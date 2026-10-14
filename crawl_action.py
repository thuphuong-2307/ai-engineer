"""
Crawl theo lô trên GitHub Actions, tự lưu kết quả lên Hugging Face.

Mỗi lượt chạy:
  1) tải danh sách URL (id, url) từ HF,
  2) đọc "trạng thái" các lượt trước trên HF (id đã xong, id lỗi vĩnh viễn, site bị chặn),
  3) crawl các URL còn lại cho tới khi hết thời gian MAX_MINUTES,
  4) cứ mỗi UPLOAD_EVERY URL xử lý xong thì đẩy 1 commit lên HF gồm:
       crawled/part-<tag>.jsonl   nội dung bài đã crawl
       state/done-<tag>.txt       id đã crawl thành công
       state/bad-<tag>.txt        id lỗi vĩnh viễn (404/410/nội dung không hợp lệ)
       state/blocked-<tag>.txt    site bị tạm dừng (liên tiếp lỗi)
       logs/errors-<tag>.txt      chi tiết lỗi

Mỗi site (nhóm domain) có hàng đợi và số worker riêng, nên không bao giờ dồn quá
SITE_LIMIT request đồng thời vào cùng một site.

Thứ tự crawl: theo đúng thứ tự dòng trong file danh sách, bắt đầu từ dòng START_ROW (mặc định 400.000),
bỏ qua các id đã xong. Mỗi site chỉ thuộc về MỘT shard (chia theo hash của site), nên số request đồng thời
vào mỗi site luôn giữ ở mức DEFAULT_SITE_LIMIT dù chạy bao nhiêu job song song.

(Tùy chọn) nạp trạng thái từ các file đã crawl trước đó:  python crawl_action.py --seed
"""
import asyncio
import datetime
import json
import logging
import os
import random
import sys
import time
import zlib
from collections import Counter, defaultdict, deque
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from bs4.dammit import EncodingDetector
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, snapshot_download
from pydantic import BaseModel, Field, ValidationError

logging.getLogger("aiohttp").setLevel(logging.ERROR)


def env(name, default):
    return os.environ.get(name, default)


# ==========================================
# 1. CẤU HÌNH (đổi qua biến môi trường trong file workflow)
# ==========================================
HF_TOKEN = (os.environ.get("HF_TOKEN") or "").strip() or None   # secret thiếu/rỗng -> None


def require_token():
    """Ghi lên Hugging Face bắt buộc phải có token; báo lỗi rõ ràng thay vì lỗi header khó hiểu."""
    if not HF_TOKEN:
        sys.exit("[LỖI] HF_TOKEN đang trống. Vào repo GitHub này: Settings > Secrets and variables > Actions > "
                 "New repository secret, tên đúng là HF_TOKEN, giá trị là token Hugging Face quyền Write. "
                 "(Secret được lưu riêng cho từng repo GitHub.)")
IN_REPO = env("HF_IN_REPO", "dhthufuong/ai-actions")          # nơi chứa danh sách URL
IN_FILE = env("HF_IN_FILE", "missing_urls.csv")              # .jsonl / .csv / .parquet có cột id, url
OUT_REPO = env("HF_OUT_REPO", "dhthufuong/ai-actions")        # nơi lưu kết quả
SEED_REPO = env("HF_SEED_REPO", OUT_REPO)                      # repo chứa các file đã crawl trước đó (chỉ dùng cho --seed)

SHARD = int(env("SHARD", "0"))                  # chạy song song nhiều job: mỗi site thuộc đúng 1 shard
NUM_SHARDS = int(env("NUM_SHARDS", "1"))
START_ROW = int(env("START_ROW", "0"))     # bắt đầu từ dòng thứ START_ROW của file (đếm từ 0)
START_ID = int(env("START_ID", "0"))            # (tùy chọn) chỉ lấy id >= START_ID; 0 = không dùng
RUN_SELECT = int(env("RUN_SELECT", "60000"))    # số URL chưa xử lý mỗi shard lấy cho 1 lượt (theo thứ tự file)
MAX_MINUTES = float(env("MAX_MINUTES", "315"))  # thời gian crawl tối đa mỗi lượt (job tối đa 360 phút)
UPLOAD_EVERY = int(env("UPLOAD_EVERY", "3000")) # số URL xử lý xong giữa 2 lần upload

GLOBAL_LIMIT = int(env("GLOBAL_LIMIT", "40"))   # tổng request đồng thời
DEFAULT_SITE_LIMIT = 2                          # request đồng thời tối đa cho mỗi site
SITE_LIMITS = {"39.net": 1}                     # site hay bị CAPTCHA: chỉ 1 request
REQUEST_DELAY = (0.5, 1.5)                      # nghỉ ngẫu nhiên trước mỗi request (giây)
MAX_RETRIES = 3
FAIL_LIMIT = 30                                 # liên tiếp lỗi bấy nhiêu lần -> dừng site

# Các domain con cùng một site (dùng chung giới hạn kết nối)
GROUPS = ("39.net", "familydoctor.com.cn", "iiyi.com", "99.com.cn",
          "vietnamplus.vn", "giadinhonline.vn")

RETRYABLE_HTTP = {"HTTP_429", "HTTP_500", "HTTP_502", "HTTP_503", "HTTP_504", "HTTP_521"}
PERMANENT = {"HTTP_404", "HTTP_410", "ValidationError"}   # ghi vào state/bad, lần sau không thử lại
NOT_COUNTED = {"HTTP_404", "HTTP_410"}                    # không tính vào chuỗi lỗi liên tiếp của site

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,zh-CN;q=0.8,zh;q=0.7,en;q=0.6",
    "Accept-Encoding": "gzip, deflate",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "no-cache",
}

ZH_DOMAINS = """
article.iiyi.com ask.39.net ask.familydoctor.com.cn baby.39.net
baidianfeng.familydoctor.com.cn bingli.iiyi.com byby.39.net cancer.39.net
care.39.net ek.39.net fitness.39.net fk.39.net fk.99.com.cn food.39.net
gan.39.net gc.39.net gk.39.net health.people.com.cn heart.39.net jb39.com
jbk.39.net man.39.net nk.39.net pf.39.net qihuangzhishu.com
qy.familydoctor.com.cn shen.39.net sj.39.net test.pmphai.com tnb.39.net
v.familydoctor.com.cn wei.39.net woman.39.net www.120ask.com www.a-hospital.com
www.baidu.com www.cnkang.com www.familydoctor.cn www.familydoctor.com.cn
www.msdmanuals.cn www.wujue.com www.youlai.cn www.zydcd.com
yanglao.familydoctor.com.cn ypk.familydoctor.com.cn zhongyibaodian.net
zysjonline.com
""".split()
ZH_SET = {d[4:] if d.startswith("www.") else d for d in ZH_DOMAINS}
LONGCHAU = {"nhathuoclongchau.com.vn", "tiemchunglongchau.com.vn"}


class MedicalDocument(BaseModel):
    doc_id: str
    url: str
    domain: str
    title: str = Field(..., min_length=2)
    content: str = Field(..., min_length=50)
    language: str = "vi"


# ==========================================
# 2. HÀM HỖ TRỢ DOMAIN
# ==========================================
def get_domain(url: str) -> str:
    return urlparse(url or "").netloc.lower().split(":")[0]


def strip_www(d: str) -> str:
    return d[4:] if d.startswith("www.") else d


def site_key(domain: str) -> str:
    """Gộp các domain con cùng một site (vd: baby.39.net, ask.39.net -> 39.net)."""
    d = strip_www(domain)
    for g in GROUPS:
        if d == g or d.endswith("." + g):
            return g
    return d


def language_of(domain: str) -> str:
    return "zh" if strip_www(domain) in ZH_SET else "vi"


# ==========================================
# 3. BÓC TÁCH NỘI DUNG (nhận bytes để tự nhận diện bảng mã, quan trọng với trang GBK)
# ==========================================
def decode_html(raw: bytes, domain: str) -> str:
    """Giải mã HTML theo thứ tự: charset khai báo trong trang -> utf-8 -> gb18030 (chỉ trang tiếng Trung).
    Chỉ khi cả ba đều lỗi mới thay ký tự hỏng bằng \ufffd."""
    cands = []
    declared = EncodingDetector.find_declared_encoding(raw, is_html=True)
    if declared:
        d = declared.lower()
        cands.append("gb18030" if d in ("gb2312", "gbk") else d)   # gb18030 là tập cha của gb2312/gbk
    cands.append("utf-8")
    if language_of(domain) == "zh":
        cands.append("gb18030")
    for enc in cands:
        try:
            return raw.decode(enc).lstrip("\ufeff")
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def extract_content(raw: bytes, domain: str) -> tuple[str, str]:
    soup = BeautifulSoup(decode_html(raw, domain), "html.parser")

    title = ""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(" ", strip=True)
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()

    parts = []
    # Long Châu: vùng thân bài xác định qua thẻ <p dir="ltr">
    if strip_www(domain) in LONGCHAU:
        first = soup.select_one('p[dir="ltr"], p[ltr]')
        if first:
            lead = soup.select_one("p.text-base.font-normal")
            if lead and lead.get_text(strip=True):
                parts.append(lead.get_text(" ", strip=True))
            for el in first.parent.find_all(["h2", "h3", "h4", "p", "li"]):
                t = el.get_text(" ", strip=True)
                if t:
                    parts.append(t)

    if len(" ".join(parts)) < 50:       # cách chung cho mọi trang
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()
        parts = [p.get_text(strip=True) for p in soup.find_all("p")
                 if len(p.get_text(strip=True)) > 30]
        if len("\n".join(parts)) < 50:
            box = soup.find("article") or soup.find("main")
            if box:
                parts = [t for t in box.get_text("\n", strip=True).split("\n") if t]

    cleaned, prev = [], None
    for t in parts:                     # bỏ dòng lặp liên tiếp
        if t != prev:
            cleaned.append(t)
        prev = t
    return title, "\n".join(cleaned)


# ==========================================
# 4. NẠP DỮ LIỆU & TRẠNG THÁI TỪ HUGGING FACE
# ==========================================
def load_input() -> pd.DataFrame:
    print(f"Đang tải danh sách URL: {IN_REPO}/{IN_FILE}")
    path = hf_hub_download(IN_REPO, IN_FILE, repo_type="dataset",
                           token=HF_TOKEN, force_download=True)
    if IN_FILE.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif IN_FILE.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_json(path, lines=True)
    if "id" not in df.columns:                  # chấp nhận tên cột id khác: doc_id, id_doc
        for alt in ("doc_id", "id_doc"):
            if alt in df.columns:
                df = df.rename(columns={alt: "id"})
                break
    if not {"id", "url"} <= set(df.columns):
        sys.exit(f"[LỖI] File {IN_FILE} cần có cột 'id' và 'url'; hiện có: {list(df.columns)}")
    df = df[["id", "url"]].copy()
    df["row"] = range(len(df))                 # vị trí dòng trong file (đếm từ 0), tính trước khi bỏ dòng thiếu
    df = df.dropna()
    df["id"] = df["id"].astype("int64")
    df["url"] = df["url"].astype(str)
    return df


def load_state():
    """Đọc các file trạng thái của mọi lượt trước. Trả về (done_ids, bad_ids, blocked_sites)."""
    done, bad, blocked = set(), set(), set()
    try:
        root = snapshot_download(OUT_REPO, repo_type="dataset", allow_patterns=["state/*"],
                                 token=HF_TOKEN)
    except Exception as e:
        print(f"[CẢNH BÁO] Không đọc được state ({type(e).__name__}); coi như chưa có.")
        return done, bad, blocked
    sdir = Path(root) / "state"
    if sdir.is_dir():
        for fp in sdir.iterdir():
            tokens = [ln.split("\t")[0].strip() for ln in fp.read_text(encoding="utf-8").splitlines()]
            if fp.name.startswith("done-"):
                done.update(int(t) for t in tokens if t.isdigit())
            elif fp.name.startswith("bad-"):
                bad.update(int(t) for t in tokens if t.isdigit())
            elif fp.name.startswith("blocked-"):
                blocked.update(t for t in tokens if t)
    print(f"Trạng thái: {len(done):,} id đã xong | {len(bad):,} id lỗi vĩnh viễn | "
          f"{len(blocked)} site bị chặn {sorted(blocked)}")
    return done, bad, blocked


def shard_of(site: str, num_shards: int) -> int:
    """Gán mỗi site cho đúng một shard (ổn định giữa các lần chạy)."""
    return zlib.crc32(site.encode("utf-8")) % num_shards


def select_candidates(df, done, bad, blocked, shard, num_shards, start_row, start_id, limit):
    """Lấy `limit` URL chưa xử lý đầu tiên (theo thứ tự file) kể từ dòng start_row, thuộc shard này."""
    df = df[df["row"] >= start_row]
    if start_id:
        df = df[df["id"] >= start_id]
    df = df[~df["id"].isin(done | bad)]
    dom = (df["url"].str.extract(r"^\s*(?:[a-zA-Z][a-zA-Z0-9+.-]*://)?([^/:?#\s]+)", expand=False)
           .str.lower())
    mapping = {d: site_key(d) for d in dom.dropna().unique()}
    df = df.assign(domain=dom, site=dom.map(mapping))
    df = df[df["site"].notna() & ~df["site"].isin(blocked)]
    site_shard = {k: shard_of(k, num_shards) for k in df["site"].unique()}
    df = df[df["site"].map(site_shard) == shard]
    remaining = len(df)
    return df.head(limit), remaining


def commit(ops, message) -> bool:
    api = HfApi(token=HF_TOKEN)
    for attempt in range(4):
        try:
            api.create_commit(repo_id=OUT_REPO, repo_type="dataset",
                              operations=ops, commit_message=message)
            return True
        except Exception as e:
            print(f"  [commit lỗi lần {attempt + 1}] {type(e).__name__}: {str(e)[:150]}")
            time.sleep(5 * (attempt + 1))
    return False


def check_write_access():
    """Thử ghi 1 file nhỏ ngay khi bắt đầu; nếu token không có quyền ghi thì dừng sớm,
    thay vì crawl hàng giờ rồi mới phát hiện không upload được."""
    now = datetime.datetime.now(datetime.timezone.utc)
    ok = commit([CommitOperationAdd(path_in_repo=f"logs/ping-s{SHARD}-{now:%Y%m%d-%H%M%S}.txt",
                                    path_or_fileobj=b"ok\n")],
                f"Kiểm tra quyền ghi (shard {SHARD})")
    if not ok:
        sys.exit(f"[LỖI] Không ghi được vào repo '{OUT_REPO}'. Token HF_TOKEN thiếu quyền Write với repo này "
                 f"(lỗi 403 nghĩa là token chỉ đọc, hoặc token fine-grained chưa được cấp quyền ghi cho repo). "
                 f"Tạo token mới có quyền Write rồi cập nhật secret HF_TOKEN trong GitHub.")
    print("Đã kiểm tra: token có quyền ghi.")


# ==========================================
# 5. CRAWL
# ==========================================
def err(url, category, etype, detail=""):
    return {"status": "error", "category": category, "type": etype,
            "msg": f"{url} | {category} | {etype} | {detail}"}


async def fetch_and_parse(session, row: dict):
    url, doc_id = row["url"], str(row["id"])
    domain = get_domain(url)
    try:
        async with session.get(url, headers=HEADERS, ssl=False) as resp:
            if resp.status != 200:
                cf = "CF" if "cf-ray" in resp.headers else ""
                return err(url, "HTTP", f"HTTP_{resp.status}",
                           f"{resp.reason} Server={resp.headers.get('Server', '')} {cf}")
            final_url = str(resp.url)
            if final_url != url and ("verify" in final_url.lower() or "captcha" in final_url.lower()):
                return err(url, "CAPTCHA", "VERIFY_PAGE", f"-> {final_url[:120]}")
            raw = await resp.read()

        title, content = await asyncio.to_thread(extract_content, raw, domain)
        doc = MedicalDocument(doc_id=doc_id, url=url, domain=domain,
                              title=title or "Không có tiêu đề", content=content,
                              language=language_of(domain))
        return {"status": "success", "data": doc.model_dump()}

    except ValidationError as ve:
        fields = sorted({str(e["loc"][0]) for e in ve.errors() if e.get("loc")})
        return err(url, "VALIDATION", "ValidationError", f"lỗi ở trường: {', '.join(fields)}")
    except asyncio.TimeoutError:
        return err(url, "TIMEOUT", "TimeoutError")
    except aiohttp.ClientError as e:
        detail = " ".join(f"{e} {getattr(e, 'os_error', '') or ''}".split())[:200]
        return err(url, "KET_NOI", type(e).__name__, detail)
    except Exception as e:
        return err(url, "KHAC", type(e).__name__, str(e)[:150])


def is_retryable(res) -> bool:
    return res["category"] in {"KET_NOI", "TIMEOUT"} or res["type"] in RETRYABLE_HTTP


class Run:
    """Trạng thái của một lượt chạy."""

    def __init__(self, deadline):
        self.deadline = deadline
        self.ok_lines, self.ok_ids, self.bad_lines, self.err_lines = [], [], [], []
        self.paused, self.new_paused = set(), set()
        self.streak = Counter()
        self.stats = Counter()
        self.n_ok = self.n_bad = self.n_err = 0
        self.n_total = 0
        self.since_flush = 0
        self.flush_no = 0
        self.lock = asyncio.Lock()

    def stop(self) -> bool:
        return time.time() >= self.deadline

    def handle(self, key, row, res):
        self.since_flush += 1
        self.n_total += 1
        if res["status"] == "success":
            self.ok_lines.append(json.dumps(res["data"], ensure_ascii=False))
            self.ok_ids.append(str(row["id"]))
            self.streak[key] = 0
            self.n_ok += 1
            return

        self.err_lines.append(res["msg"])
        self.stats[(res["category"], res["type"])] += 1
        if res["type"] in PERMANENT:
            self.bad_lines.append(f"{row['id']}\t{res['type']}")
            self.n_bad += 1
        else:
            self.n_err += 1
        if res["type"] not in NOT_COUNTED:
            self.streak[key] += 1
            if self.streak[key] >= FAIL_LIMIT and key not in self.paused:
                self.paused.add(key)
                self.new_paused.add(key)
                print(f"  [TẠM DỪNG] {key}: {FAIL_LIMIT} URL liên tiếp lỗi, bỏ qua phần còn lại.")


async def flush(R: Run, force=False):
    async with R.lock:
        if not force and R.since_flush < UPLOAD_EVERY:
            return
        if not (R.ok_lines or R.bad_lines or R.err_lines or R.new_paused):
            return
        ok_lines, ok_ids, bad, errs, newp = (R.ok_lines, R.ok_ids, R.bad_lines,
                                             R.err_lines, R.new_paused)
        R.ok_lines, R.ok_ids, R.bad_lines, R.err_lines, R.new_paused = [], [], [], [], set()
        R.since_flush = 0
        R.flush_no += 1
        now = datetime.datetime.now(datetime.timezone.utc)
        tag = f"{now:%Y%m%d-%H%M%S}-s{SHARD}-{R.flush_no}"

        ops = []

        def add(path, lines):
            if lines:
                ops.append(CommitOperationAdd(path_in_repo=path,
                                              path_or_fileobj=("\n".join(lines) + "\n").encode("utf-8")))

        add(f"crawled/part-{tag}.jsonl", ok_lines)
        add(f"state/done-{tag}.txt", ok_ids)
        add(f"state/bad-{tag}.txt", bad)
        add(f"state/blocked-{tag}.txt", sorted(newp))
        add(f"logs/errors-{tag}.txt", errs)

        ok = await asyncio.to_thread(commit, ops,
                                     f"Crawl shard {SHARD}: +{len(ok_lines)} bài ({now:%Y-%m-%d %H:%M} UTC)")
        if ok:
            print(f"  [UPLOAD] {tag}: {len(ok_lines)} bài, {len(bad)} lỗi vĩnh viễn, {len(errs)} dòng lỗi")
        else:
            print("  [UPLOAD LỖI] giữ lại dữ liệu, sẽ thử lại ở lần flush sau")
            R.ok_lines = ok_lines + R.ok_lines
            R.ok_ids = ok_ids + R.ok_ids
            R.bad_lines = bad + R.bad_lines
            R.err_lines = errs + R.err_lines
            R.new_paused |= newp


async def worker(key, dq, session, gsem, R: Run):
    while dq and not R.stop() and key not in R.paused:
        doc_id, url = dq.popleft()
        row = {"id": doc_id, "url": url}
        await asyncio.sleep(random.uniform(*REQUEST_DELAY))

        res = None
        for attempt in range(MAX_RETRIES + 1):
            if R.stop() or key in R.paused:
                break
            async with gsem:
                res = await fetch_and_parse(session, row)
            if res["status"] == "success" or not is_retryable(res) or attempt == MAX_RETRIES:
                break
            await asyncio.sleep(2 ** attempt + random.random())

        if res is None:
            break
        R.handle(key, row, res)
        if R.since_flush >= UPLOAD_EVERY:
            await flush(R)


async def reporter(R: Run, t0: float):
    while True:
        await asyncio.sleep(300)
        h = (time.time() - t0) / 3600
        print(f"[{h:.1f}h] thành công {R.n_ok:,} | lỗi vĩnh viễn {R.n_bad:,} | lỗi khác {R.n_err:,} "
              f"| ~{R.n_ok / max(h, 1e-9):,.0f} bài/giờ | site tạm dừng: {len(R.paused)}")


async def crawl():
    require_token()
    check_write_access()
    t0 = time.time()
    df = load_input()
    print(f"Danh sách URL: {len(df):,} dòng")

    done, bad, blocked = load_state()
    cand, remaining = select_candidates(df, done, bad, blocked, SHARD, NUM_SHARDS,
                                        START_ROW, START_ID, RUN_SELECT)
    print(f"Bắt đầu từ dòng {START_ROW:,}. Shard {SHARD}/{NUM_SHARDS}: còn {remaining:,} URL chưa xử lý; "
          f"lượt này lấy {len(cand):,} URL từ {cand['site'].nunique() if len(cand) else 0} site")
    if len(cand):
        top = cand["site"].value_counts().head(5)
        print("Site nhiều URL nhất trong lượt này:", ", ".join(f"{k} ({v:,})" for k, v in top.items()))
    if cand.empty:
        print("Không còn URL nào để crawl.")
        return

    by_site = defaultdict(deque)
    for r in cand[["id", "url", "site"]].itertuples(index=False):
        by_site[r.site].append((int(r.id), r.url))

    R = Run(deadline=t0 + MAX_MINUTES * 60)
    connector = aiohttp.TCPConnector(limit=GLOBAL_LIMIT, limit_per_host=4, ttl_dns_cache=300, ssl=False)
    timeout = aiohttp.ClientTimeout(total=40, sock_read=20, sock_connect=10)

    rep = None
    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            gsem = asyncio.Semaphore(GLOBAL_LIMIT)
            rep = asyncio.create_task(reporter(R, t0))
            tasks = []
            for key, dq in by_site.items():
                for _ in range(min(SITE_LIMITS.get(key, DEFAULT_SITE_LIMIT), len(dq))):
                    tasks.append(worker(key, dq, session, gsem, R))
            await asyncio.gather(*tasks)
    finally:
        # Chạy cả khi bị hủy giữa chừng (Cancel workflow / Ctrl+C): đẩy nốt phần dữ liệu còn trong bộ nhớ
        if rep:
            rep.cancel()
        await flush(R, force=True)

    hours = (time.time() - t0) / 3600
    rate = R.n_ok / max(hours, 1e-9)
    left = remaining - R.n_ok - R.n_bad
    print("\n===== TỔNG KẾT LƯỢT NÀY =====")
    print(f"Đã thử: {R.n_total:,} URL (shard {SHARD}/{NUM_SHARDS}) | Thành công: {R.n_ok:,} | "
          f"Lỗi vĩnh viễn: {R.n_bad:,} | Lỗi khác: {R.n_err:,} | Thời gian: {hours:.1f}h | "
          f"Tốc độ: ~{rate:,.0f} bài/giờ")
    if R.paused:
        print("Site bị tạm dừng:", ", ".join(sorted(R.paused)))
    for (cat, typ), n in R.stats.most_common(15):
        print(f"  {cat:<12} {typ:<22} {n:,}")
    print(f"Còn chưa có kết quả (ước tính): {left:,} URL")


# ==========================================
# 6. NẠP TRẠNG THÁI TỪ FILE ĐÃ CRAWL TRƯỚC ĐÓ (chạy 1 lần)
# ==========================================
def seed():
    require_token()
    files = [f.strip() for f in env("SEED_FILES", "").split(",") if f.strip()]
    if not files:
        print("SEED_FILES đang trống nên không có gì để seed (file missing_urls.csv đã loại sẵn các URL đã crawl).")
        return
    ops = []
    for fn in files:
        try:
            path = hf_hub_download(SEED_REPO, fn, repo_type="dataset", token=HF_TOKEN, force_download=True)
        except Exception as e:
            print(f"[LỖI] Không tải được '{fn}' từ repo '{SEED_REPO}': {type(e).__name__}")
            try:
                from huggingface_hub import list_repo_files
                print(f"Các file đang có trong '{SEED_REPO}':")
                for name in list_repo_files(SEED_REPO, repo_type="dataset", token=HF_TOKEN):
                    print("  -", name)
            except Exception as e2:
                print(f"Cũng không liệt kê được file của repo: {type(e2).__name__}")
            sys.exit("-> Sửa SEED_FILES (và HF_SEED_REPO) trong file workflow cho đúng tên file rồi chạy lại.")
        ids = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                d = rec.get("doc_id", rec.get("id"))
                if d is not None and str(d).isdigit():
                    ids.add(int(d))
        print(f"{fn}: {len(ids):,} id")
        ops.append(CommitOperationAdd(path_in_repo=f"state/done-seed-{Path(fn).stem}.txt",
                                      path_or_fileobj="\n".join(map(str, sorted(ids))).encode("utf-8")))
    if not ops:
        sys.exit("Không có file nào để seed (kiểm tra SEED_FILES).")
    if not commit(ops, "Seed state từ các file đã crawl"):
        sys.exit("[LỖI] Không upload được state seed lên Hugging Face (kiểm tra HF_TOKEN có quyền Write và tên HF_OUT_REPO).")
    print("Đã lưu state seed.")


if __name__ == "__main__":
    if "--seed" in sys.argv:
        seed()
    else:
        asyncio.run(crawl())
