# L2S-JSSP: Một Framework Học Tăng Cường Hướng Tìm Kiếm Cục Bộ trên Đồ Thị Phân Tách cho Bài toán Lập Lịch Job-Shop

### *L2S-JSSP: A Reinforcement-Learning Guided Local-Search Framework over the Disjunctive Graph for the Job-Shop Scheduling Problem*

[![CI](https://gitlab.com/thanhdo2404-group/demo-jssp-model/badges/improve-l2s-jssp/pipeline.svg)](https://gitlab.com/thanhdo2404-group/demo-jssp-model/-/pipelines)

> **Reference base** — Cong Zhang et al., *Deep Reinforcement Learning Guided Improvement Heuristic for Job Shop Scheduling*, ICLR 2024. Framework này là bản tái hiện từ đầu (*from-scratch*) bằng **PyTorch**, mở rộng với multi-head attention, GPU message-passing hai chiều (EST/LST), actor-critic và curriculum learning, kèm đánh giá toàn diện trên các bộ benchmark kinh điển.

---

## Abstract / Tóm tắt

**(EN)** The Job-Shop Scheduling Problem (JSSP) is a canonical NP-hard combinatorial optimization problem. We present **L2S-JSSP**, a single-file, Colab-runnable framework that learns an *improvement* heuristic: instead of constructing a schedule operation-by-operation, the agent starts from a complete feasible solution and *learns to search* its neighbourhood. The state is the complete-solution **disjunctive graph**; the action is an operation-pair swap inside a **critical block** (the N5 neighbourhood); the reward is the per-step improvement of the best makespan found. The policy is a Graph Neural Network combining a **GIN-based Topological Embedding Module (TPM)** and a **multi-head GAT-based Context-Aware Module (CAM)**, trained by **n-step advantage actor-critic**. Node features (earliest/latest start times) are produced by a **batched GPU message-passing operator** running both a forward (EST) and a backward (LST) pass over the dense adjacency matrix. We benchmark against five classic heuristics — **SPT, LPT, FDD, MWKR** (non-delay dispatching) and **SBT** (Shifting Bottleneck Heuristic) — on FT, LA, ABZ and TA instances. With multi-sample stochastic rollouts, the learned policy outperforms the best heuristic on 8 of 11 instances despite being trained only on 6×6 random instances, demonstrating strong size-agnostic transfer.

**(VI)** Bài toán Lập lịch Job-Shop (JSSP) là bài toán tối ưu tổ hợp NP-khó kinh điển. Chúng tôi trình bày **L2S-JSSP**, một framework đơn tập tin, chạy được trên Colab, học một heuristic *cải thiện* (*improvement heuristic*): thay vì xây dựng lịch từng thao tác, tác nhân xuất phát từ một lời giải khả thi hoàn chỉnh và *học cách tìm kiếm* trong lân cận của nó. Trạng thái là **đồ thị phân tách** (*disjunctive graph*) của lời giải hoàn chỉnh; hành động là phép hoán đổi một cặp thao tác trong **khối tới hạn** (*critical block*, lân cận N5); phần thưởng là mức cải thiện makespan tốt nhất theo từng bước. Mạng chính sách là một GNN kết hợp **module nhúng tô-pô dựa trên GIN (TPM)** và **module nhận thức ngữ cảnh dựa trên GAT đa đầu (CAM)**, huấn luyện bằng **actor-critic n-bước**. Đặc trưng nút (thời điểm bắt đầu sớm/muộn nhất) được tính bằng **toán tử truyền thông điệp GPU theo lô** chạy cả chiều xuôi (EST) và chiều ngược (LST). Chúng tôi so sánh với năm heuristic kinh điển — **SPT, LPT, FDD, MWKR** và **SBT** — trên các bộ FT, LA, ABZ, TA. Với lấy mẫu ngẫu nhiên đa lần, chính sách học được vượt qua heuristic tốt nhất trên 8/11 thể hiện dù chỉ được huấn luyện trên thể hiện 6×6 ngẫu nhiên, chứng tỏ khả năng chuyển giao bất biến theo kích thước.

**Keywords:** Job-Shop Scheduling, Reinforcement Learning, Graph Neural Networks, Local Search, Disjunctive Graph, Critical Path Method, Actor-Critic.

---

## 1. Introduction / Giới thiệu

Bài toán JSSP yêu cầu lập lịch $J$ công việc (*jobs*) trên $M$ máy, mỗi công việc gồm một chuỗi thao tác (*operations*) có thứ tự cố định, mỗi thao tác chạy trên một máy xác định trong một khoảng thời gian cho trước, sao cho **makespan** (thời điểm hoàn thành công việc cuối cùng, $C_{\max}$) là nhỏ nhất. Đây là bài toán NP-khó điển hình.

Các cách tiếp cận học sâu trước đây thường theo hướng *construction* (xây dựng lời giải từng thao tác). Ngược lại, **Learning-to-Search (L2S)** theo hướng *improvement*: bắt đầu từ một lời giải hoàn chỉnh và lặp lại việc áp dụng các phép di chuyển cục bộ (*local moves*) để giảm dần makespan. Đóng góp của framework này:

1. **Tái hiện đầy đủ 5 trụ cột của L2S** trong một notebook duy nhất, chỉ dùng `torch`, `numpy`, `matplotlib`.
2. **GPU message-passing hai chiều** tính cả EST (xuôi) và LST (ngược) thay cho vòng lặp NumPy tuần tự.
3. **Multi-head attention** trong module CAM.
4. **Actor-critic n-bước** thay cho REINFORCE thuần, giảm phương sai gradient.
5. **Curriculum learning** tinh chỉnh tuần tự trên thể hiện lớn dần.
6. **Đánh giá đối sánh** với 5 heuristic cổ điển trên benchmark kinh điển, kèm **multi-sample evaluation**.

---

## 2. Files / Cấu trúc tập tin

| File | Vai trò |
|------|---------|
| `L2S_JSSP_Colab.ipynb` | **Sản phẩm chính.** 19 cell: data → graph/CPM → GPU MP-EST/LST → N5 → env → GIN + multi-head GAT actor-critic → n-step training → curriculum fine-tuning → benchmark evaluation → Gantt. Mở trong Colab và Run-All. |
| `L2S_JSSP_Colab_result.ipynb` | Phiên bản đã chạy toàn bộ trên Colab (T4 GPU) kèm output và hình ảnh; nguồn của các kết quả thực nghiệm ở Mục 6. |
| `build_notebook.py` | Trình sinh notebook (sửa ở đây rồi chạy `python3 build_notebook.py`). Job CI `build-notebook` tự dựng lại notebook như artifact mỗi lần push. |
| `_validate_core.py` | Kiểm chứng thuần Python (không phụ thuộc) của lõi thuật toán — 6 nhóm kiểm tra, chạy trong CI (job `validate-core`). |

---

## 3. Methodology / Phương pháp

Pipeline được trình bày theo đúng thứ tự các cell trong codebase.

### 3.1. Biểu diễn thể hiện (Cell 4)

Một thể hiện được lưu bởi lớp `JSSPInstance`:
- `times[j][i]`: thời gian xử lý thao tác thứ $i$ của công việc $j$.
- `machines[j][i]`: máy xử lý thao tác đó (mỗi công việc đi qua mọi máy đúng một lần).
- Chỉ số thao tác toàn cục: $op = j \cdot M + i$; nguồn giả $S = J\cdot M$, đích giả $T = J\cdot M + 1$.
- Các cung **conjunctive** (ràng buộc thứ tự công việc) **không đổi** trong quá trình tìm kiếm nên được **tính trước một lần** dưới dạng mảng chỉ số (cho ma trận kề) và danh sách tiền/hậu nhiệm (cho CPM).

Bộ phân tích `parse_standard_jssp` đọc định dạng chuẩn FT/LA/ABZ/ORB/SWV/TA/YN: dòng đầu `J M`, mỗi dòng sau là các cặp `machine time` (máy đánh chỉ số từ 0).

### 3.2. Đồ thị phân tách và đánh giá lịch (Cell 5)

Một *lời giải* là thứ tự xử lý các thao tác trên mỗi máy (`machine_seq[m]`). Từ đó dựng DAG gồm cung conjunctive (cố định) và cung **disjunctive** (theo thứ tự máy, phụ thuộc lời giải).

**(a) Critical Path Method — `cpm_schedule`.** Sắp thứ tự tô-pô rồi tính:
- EST (xuôi): $\mathrm{EST}(v) = \max_{u \in \mathrm{pred}(v)} \big(\mathrm{EST}(u) + p(u)\big)$,
- makespan $= \mathrm{EST}(T)$,
- LST (ngược): $\mathrm{LST}(u) = \min_{v \in \mathrm{succ}(u)} \mathrm{LST}(v) - p(u)$.

CPM là bộ đánh giá chuẩn, dùng chung cho **mọi phương pháp** được so sánh (qua `makespan_of`), đảm bảo công bằng.

**(b) GPU message-passing hai chiều — `mp_est_lst_gpu`.** Đây là trụ cột số 5, chạy trên GPU bằng `torch` matmul dầy đặc, không vòng lặp Python trên nút:
- **Xuôi (EST):** $d \leftarrow \max\big(A_{\text{all}} \odot (p + d)^\top$ theo chiều tiền nhiệm$,\ d\big)$, hội tụ sau tối đa $N$ vòng (dừng sớm khi $\Delta < \epsilon$).
- **Ngược (LST):** chạy cùng toán tử trên đồ thị đảo ($A_{\text{all}}^\top$) với giá trị $-\mathrm{lft}$ để lan truyền thời điểm hoàn thành muộn nhất ngược từ $T$; sau đó $\mathrm{LST}(u) = \mathrm{lft}(u) - p(u)$.

Kết quả (`est_mp`, `lst_mp`) được dùng làm đặc trưng nút trong trạng thái. CPM vẫn được giữ cho kiểm tra tính khả thi (`topo_order` ném lỗi nếu có chu trình) và cho phép đi đường găng (cần EST/LST chính xác).

### 3.3. Đường găng, khối tới hạn và lân cận N5 (Cell 6)

- `find_critical_path`: đi một đường $S \to T$ qua các cạnh *khít* (tight) và *tới hạn* ($\mathrm{EST} = \mathrm{LST}$).
- `critical_blocks`: gom các thao tác liên tiếp trên đường găng cùng một máy thành một khối.
- `n5_candidate_moves`: lân cận **N5** — hoán đổi cặp đầu/cuối trong mỗi khối tới hạn; khối đầu chỉ lấy cặp cuối, khối cuối chỉ lấy cặp đầu; $|N5| \le 2N(s) - 2$.

### 3.4. Lời giải khởi tạo và các baseline heuristic (Cell 7)

Tất cả đều được chấm điểm bằng cùng bộ đánh giá CPM:

| Ký hiệu | Loại | Quy tắc ưu tiên |
|--------|------|----------------|
| **FDD/MWKR** | khởi tạo cho L2S | tỉ lệ nhỏ nhất Flow-Due-Date / Most-Work-Remaining |
| **SPT** | non-delay dispatching | thời gian xử lý ngắn nhất trước |
| **LPT** | non-delay dispatching | thời gian xử lý dài nhất trước |
| **FDD** | non-delay dispatching | thời gian tích lũy (Flow Due Date) nhỏ nhất |
| **MWKR** | non-delay dispatching | công việc còn nhiều việc nhất |
| **SBT** | Shifting Bottleneck Heuristic | xác định máy cổ chai rồi giải bài toán 1-máy (Lmax) bằng Schrage EDD; lặp tới khi mọi máy được xếp |

### 3.5. Môi trường MDP (Cell 8)

- **State:** đồ thị lời giải hoàn chỉnh + đặc trưng nút $(p, \mathrm{est}, \mathrm{lst})$ chuẩn hóa. Ma trận kề conjunctive $A_J$ được dựng **một lần**/môi trường; mỗi bước chỉ dựng lại $A_M$ (disjunctive) bằng scatter numpy vector hóa.
- **Action:** một cặp $(u, v)$ thuộc N5 để hoán đổi.
- **Reward:** $\max(\text{best\_makespan} - \text{new\_makespan},\ 0)$.
- **Done:** hết horizon hoặc không còn nước đi N5 (trạng thái hấp thụ).

### 3.6. Mạng chính sách: GIN (TPM) + multi-head GAT (CAM) + actor + critic (Cell 10)

- **TPM (GIN):** chồng $K$ lớp GIN, $h' = \mathrm{MLP}\big((1+\epsilon)h + A h\big)$; nhúng tổng hợp là tổng embedding qua mọi lớp.
- **CAM (multi-head GAT):** hai nhánh attention đa đầu riêng cho đồ thị công việc $G_J$ và đồ thị máy $G_M$. Mỗi đầu $h$ tính $\alpha_h = \mathrm{softmax}(\mathrm{LeakyReLU}(a_{\text{src}} + a_{\text{dst}}^\top))$ trên ma trận kề có self-loop, output $\mathrm{ELU}(\alpha_h W_h h)$, nối (concat) các đầu rồi chiếu qua $W_{\text{out}}$ (mặc định `gat_heads=4`).
- **Actor head:** ghép embedding nút và đồ thị ($4d$) → MLP → vector điểm $q$ chiều; ma trận điểm cặp $SC = h_p h_p^\top$, lấy logits cho các cặp ứng viên.
- **Critic head:** đọc embedding đồ thị gom ($2d$) → giá trị vô hướng $V(s)$ làm baseline học được.

### 3.7. Huấn luyện actor-critic n-bước (Cell 12)

Trung thành với Algorithm 1 của L2S (cập nhật sau mỗi $n$ bước), nhưng baseline trung bình được thay bằng critic học được:
$$\mathcal{L} = -\sum_t \log \pi(a_t|s_t)\,(G_t - V(s_t)) + c_v \sum_t \big(V(s_t) - G_t\big)^2 - c_e \sum_t \mathcal{H}(\pi),$$
với $G_t$ là return n-bước (reward được chuẩn hóa theo makespan ban đầu), $c_v$ trọng số critic, $c_e$ hệ số entropy, kèm gradient clipping. Có `save_checkpoint`/`load_checkpoint` (`l2s_jssp_policy.pt`).

### 3.8. Curriculum learning (Cell 13b)

Do GNN bất biến theo kích thước, chính sách 6×6 được tinh chỉnh tuần tự (đặt `RUN_CURRICULUM=True`):

| Stage | Kích thước | horizon | batch | iters | Checkpoint |
|-------|-----------|---------|-------|-------|------------|
| 1 | 15×15 | 80 | 4 | 60 | `l2s_jssp_policy_15x15.pt` |
| 2 | 20×20 | 96 | 4 | 50 | `l2s_jssp_policy_20x20.pt` |
| 3 | 30×30 | 128 | 2 | 30 | `l2s_jssp_policy_30x30.pt` |
| 4 | 50×20 | 128 | 2 | 30 | `l2s_jssp_policy_50x20.pt` |

### 3.9. Suy luận: greedy và multi-sample (Cell 16)

`solve(policy, inst, steps, greedy, n_samples, seeds)`:
- `greedy=True`: một lần rollout xác định.
- `greedy=False, n_samples=k`: chạy $k$ rollout ngẫu nhiên với seed khác nhau, **lấy makespan tốt nhất**. Đây là cách hiệu quả nhất để cải thiện trên thể hiện lớn mà không cần huấn luyện lại.

---

## 4. Pillar-to-Cell Mapping / Ánh xạ trụ cột – cell

| Trụ cột L2S | Cell | Ghi chú |
|-------------|------|---------|
| Vòng lặp tìm kiếm cục bộ | `Environment` (Cell 8) | |
| Lân cận N5 (critical blocks) | Cell 6 | |
| Nhúng đồ thị: GIN (TPM) | Cell 10 | tổng $K$ lớp GIN |
| Nhúng đồ thị: multi-head GAT (CAM) | Cell 10 | `gat_heads=4`; concat → project |
| MDP (state = đồ thị lời giải, reward = cải thiện) | Cell 8 | |
| GPU MP-EST (xuôi) + MP-LST (ngược) | Cell 5 | đặc trưng nút |
| Huấn luyện actor-critic n-bước | Cell 12 | advantage = return − V(s) |
| Curriculum fine-tuning | Cell 13b | `RUN_CURRICULUM=True` |

---

## 5. Experimental Setup / Thiết lập thực nghiệm

**Môi trường chạy (từ `L2S_JSSP_Colab_result.ipynb`):** Google Colab, Python **3.12.13**, PyTorch **2.11.0+cu128**, NumPy **2.0.2**, thiết bị **CUDA (T4 GPU)**.

**Cấu hình huấn luyện mặc định (Cell 13):** `n_jobs=6, n_machines=6, horizon_T=64, n_step=8, batch_size=8, n_iterations=120, lr=5e-4, embed_dim=64, n_layers=3, gat_heads=4`. Chính sách chỉ được huấn luyện trên thể hiện **6×6 ngẫu nhiên** rồi áp dụng trực tiếp cho mọi benchmark (chuyển giao bất biến theo kích thước).

**Xác minh định dạng dữ liệu (Cell 14):** phân tích `ft06` (tối ưu đã biết = 55), kiểm tra shape `(6,6)`, mỗi hàng `machines` là hoán vị của $0..5$, makespan FDD/MWKR $\ge 55$ — xác nhận bố cục dữ liệu nhất quán với quy ước `times[j][i]` / `machines[j][i]`.

**Ngân sách suy luận:** mỗi thể hiện có (số bước L2S, số mẫu); SBT bị bỏ qua trên thể hiện $> 30$ công việc để đảm bảo thời gian chạy Colab Free Tier.

---

## 6. Results / Kết quả thực nghiệm

Bảng 1 — makespan trên các benchmark kinh điển (số nhỏ hơn là tốt hơn). Cột **L2S-g** = greedy rollout, **L2S-s** = best-of-$k$ stochastic sampling. **gap%** = phần trăm chênh lệch của L2S-s so với heuristic tốt nhất (âm = L2S tốt hơn mọi heuristic). **opt** = tối ưu đã biết. **t** = thời gian chạy cụm đó.

| instance | size | SPT | LPT | FDD | MWKR | SBT | **L2S-g** | **L2S-s** | gap% | opt | t |
|----------|------|-----|-----|-----|------|-----|-----------|-----------|------|-----|---|
| ft06 | 6×6 | 88 | 77 | 70 | 61 | 65 | 61 | **55** | −9.8 | 55 | 24.8s |
| ft10 | 10×10 | 1074 | 1295 | 1090 | 1108 | 1252 | 1242 | **1021** | −4.9 | 930 | 26.8s |
| ft20 | 20×5 | 1267 | 1631 | 1294 | 1501 | 1408 | 1696 | 1312 | +3.6 | 1165 | 29.1s |
| la01 | 10×5 | 751 | 822 | 748 | 735 | 807 | 740 | **666** | −9.4 | 666 | 25.5s |
| la16 | 10×10 | 1156 | 1229 | 1062 | 1054 | 1164 | 1156 | **988** | −6.3 | 945 | 21.7s |
| la21 | 15×10 | 1324 | 1451 | 1253 | 1264 | 1303 | 1488 | **1121** | −10.5 | 1046 | 15.4s |
| abz5 | 10×10 | 1352 | 1586 | 1363 | 1369 | 1421 | 1395 | **1284** | −5.0 | 1234 | 21.8s |
| ta01 | 15×15 | 1462 | 1701 | 1439 | 1491 | 1827 | 1722 | **1377** | −4.3 | 1231 | 12.7s |
| la31 | 30×10 | 1951 | 2245 | 1895 | 1931 | 2638 | 2163 | **1847** | −2.5 | 1784 | 7.9s |
| ta31 | 30×15 | 2335 | 2417 | 2240 | 2134 | 3203 | 2467 | 2206 | +3.4 | 1764 | 7.8s |
| ta51 | 50×15 | 3856 | 3880 | 3273 | 3435 | n/a | 4004 | 3424 | +4.6 | 2760 | 5.1s |

### 6.1. Phân tích chính

1. **L2S-sample thắng heuristic tốt nhất trên 8/11 thể hiện** (gap% âm): ft06, ft10, la01, la16, la21, abz5, ta01, la31. Mức cải thiện lớn nhất đạt **−10.5%** trên la21 (15×10).
2. **Đạt tối ưu tuyệt đối** trên **ft06 (55)** và **la01 (666)** — chứng minh cả pipeline (parse, CPM, N5, chính sách) đúng đắn.
3. **Vai trò của multi-sample rất rõ rệt:** ví dụ ft06 từ L2S-g $=61$ xuống L2S-s $=55$; ft10 từ $1242 \to 1021$; la21 từ $1488 \to 1121$. Greedy dễ kẹt cực tiểu cục bộ, sampling thoát khỏi bẫy đó.
4. **Suy giảm trên thể hiện rất lớn:** ft20 (+3.6%), ta31 (+3.4%), ta51 (+4.6%) — L2S thua heuristic. Nguyên nhân: chính sách chỉ huấn luyện trên 6×6, ngân sách bước bị cắt mạnh (100–120) và số mẫu giảm (1–2) trên thể hiện lớn. Đây chính là động lực cho **curriculum learning** (Mục 3.8).
5. **SBT không vượt trội** và tốn kém: trên ta01/ta31 cho makespan xấu hơn cả dispatching đơn giản (1827, 3203), và bị bỏ qua trên ta51 (50 công việc) vì quá chậm.
6. **MWKR là heuristic mạnh nhất trong nhóm dispatching** trên nhiều thể hiện (ft06=61, ta31=2134), nhưng vẫn bị L2S-sample vượt trên phần lớn thể hiện vừa.

### 6.2. Trực quan hóa (trong notebook kết quả)

- **Cell 15:** heatmap ma trận thời gian xử lý và ma trận thứ tự máy của ft06, kèm Gantt của lịch FDD/MWKR ban đầu.
- **Cell 17:** biểu đồ cột nhóm so sánh mọi phương pháp, có đường tối ưu đã biết.
- **Cell 18:** Gantt của lịch tốt nhất do L2S tìm được trên ft06 (makespan = 55, trùng tối ưu).

---

## 7. Reproducibility & CI / Tái lập và kiểm thử liên tục

Job CI `validate-core` chạy `python3 l2s_jssp/_validate_core.py` (thuần Python, không phụ thuộc, nhanh) mỗi lần push, gồm 6 nhóm kiểm tra:

| # | Kiểm tra | Đảm bảo |
|---|---------|---------|
| 1 | CPM + MP-EST (xuôi) | `mp_est()` bằng CPM EST trên 3×3 |
| 2 | MP-LST (ngược) | `mp_lst()` bằng CPM LST trên 3×3, 6×6, 10×10 |
| 3 | Đường găng / khối / N5 | đường đi bắt đầu tại S, kết thúc tại T; khối không rỗng |
| 4 | N5 random local search | `best_mk <= init_mk` từ 3×3 → 15×10 |
| 5 | SPT / LPT / FDD / MWKR | lịch khả thi (không chu trình) + makespan $\ge$ cận dưới |
| 6 | SBT | lịch khả thi + makespan $\ge$ cận dưới |

Job CI `build-notebook` tự dựng lại `L2S_JSSP_Colab.ipynb` từ `build_notebook.py` và lưu như artifact.

**Cách chạy:** tải `L2S_JSSP_Colab.ipynb` lên [Google Colab](https://colab.research.google.com/), chọn `Runtime → GPU` (khuyến nghị T4), rồi `Run all`. Để chạy các cell benchmark, hoặc clone repo này để có thư mục `datasets/`, hoặc làm theo hướng dẫn clone trong notebook (repo private: dùng personal access token).

Phụ thuộc: chỉ `torch`, `numpy`, `matplotlib` (đã cài sẵn trên Colab); xem `requirements.txt` cho chạy cục bộ.

---

## 8. Design Decisions vs. the Paper / So sánh thiết kế với paper gốc

| Khía cạnh | Framework này | Paper gốc |
|----------|---------------|-----------|
| Ma trận kề GNN | Dầy đặc $(N\times N)$ numpy/torch | Thưa (`torch_geometric`) |
| Ma trận kề conjunctive | Tính trước một lần/env | Dựng lại mỗi bước |
| Attention CAM | **GAT đa đầu** (`gat_heads=4`) | GAT đơn đầu |
| Đặc trưng nút | **GPU MP-EST + MP-LST** (theo lô) | MP-EST NumPy tuần tự |
| Thuật toán huấn luyện | **Actor-critic n-bước** (V(s) baseline) | REINFORCE n-bước |
| Entropy bonus | Có (ổn định) | Không |
| Curriculum | Tùy chọn (Cell 13b) | Kích thước cố định |
| Checkpointing | `l2s_jssp_policy.pt` | Không đề cập |
| Baseline so sánh | SPT, LPT, FDD, MWKR, **SBT** + multi-sample | — |

---

## 9. Limitations & Future Work / Hạn chế và hướng phát triển

1. **Khả năng mở rộng (scalability):** ma trận kề dầy đặc $(N\times N)$ giới hạn kích thước; hướng tiếp theo là sparse message-passing (`torch_geometric`) cho thể hiện 100×20.
2. **Suy giảm trên thể hiện lớn:** cần curriculum đầy đủ (Cell 13b) hoặc huấn luyện trực tiếp ở kích thước lớn, tăng ngân sách bước/số mẫu khi có GPU mạnh hơn.
3. **Cải tiến theo paper:** tăng `embed_dim=128, gat_heads=8, n_layers=4, horizon_T=500, batch_size=64, n_iterations=2000, lr=5e-5`.

---

## 10. Conclusion / Kết luận

L2S-JSSP cung cấp một bản tái hiện đầy đủ, chạy được và có thể kiểm chứng của heuristic cải thiện L2S cho JSSP, bổ sung multi-head attention, GPU message-passing hai chiều, actor-critic và curriculum learning. Trên 11 benchmark kinh điển, chính sách (chỉ huấn luyện 6×6) với multi-sample vượt qua heuristic tốt nhất trên 8/11 thể hiện và đạt tối ưu tuyệt đối trên ft06 và la01, khẳng định tiềm năng của cách tiếp cận học-hướng-tìm-kiếm. Các hạn chế trên thể hiện lớn chỉ ra hướng nghiên cứu rõ ràng về sparse GNN và curriculum.

---

## References / Tài liệu tham khảo

1. C. Zhang et al., *Deep Reinforcement Learning Guided Improvement Heuristic for Job Shop Scheduling*, ICLR 2024.
2. K. Xu et al., *How Powerful are Graph Neural Networks?* (GIN), ICLR 2019.
3. P. Veličković et al., *Graph Attention Networks* (GAT), ICLR 2018.
4. J. Adams, E. Balas, D. Zawack, *The Shifting Bottleneck Procedure for Job Shop Scheduling*, Management Science, 1988.
5. H. Fisher, G. L. Thompson (FT), J. F. Muth, G. L. Thompson, *Industrial Scheduling*, 1963.
6. S. Lawrence (LA), 1984; J. Adams et al. (ABZ); É. Taillard (TA), *Benchmarks for basic scheduling problems*, EJOR, 1993.
