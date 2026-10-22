# AMR D200｜Isaac Sim USD 模型與移動控制

**依據：使用者上傳的 `amr_d200.stl`。版本：STL 6 吋輪組版。**  
**狀態：已建立幾何與物理關節；通過交付環境的離線檢查，尚未在 OpenUSD／Isaac Sim 執行驗證。**

這個包不是單純的 STL 外觀轉檔：USD 已寫入剛體、質量與慣量、碰撞形狀、左右驅動輪關節、被動腳輪關節與輪速驅動。控制程式只改變輪關節的目標速度，沒有直接移動根節點或設定車身速度來假裝輪子在驅動。

## 1. 先注意：截圖與 STL 不是相同輪徑版本

| 項目 | 截圖 | 實際讀取 STL／本次 USD |
|---|---|---|
| 驅動輪直徑 | 8 吋，203.2 mm | **6 吋，152.4 mm** |
| 下層主車身底面高度 | 101.6 mm | **約 55 mm** |
| 主車身長 × 寬 | 950 × 650 mm | 約 950 × 650 mm |
| 含突出件的整體範圍 | 未由截圖推定 | 約 **968 × 654 × 251.5 mm** |

本次優先保留 STL 的尺寸、輪心與裝配位置，沒有擅自把整台等比放大，也沒有只放大輪胎導致穿模。STL 本身沒有單位欄位；本次依 0.95 × 0.65 的車身數值與截圖標稱尺寸，將 STL 數值解讀為公尺。要改成截圖的 8 吋版本，需要另行調整輪心、車身底面與腳輪高度，不應只改控制器的輪半徑。

## 2. 檔案怎麼選

| 檔案 | 用途 |
|---|---|
| `AMR_D200.usd` | **主模型**。15 個剛體、14 個旋轉關節。包含真正可轉向的四組雙輪腳輪。輪速預設為 0；不含地面或 PhysicsScene，供加入既有環境。 |
| `AMR_D200_drive_demo.usd` | **先開這個測試**。主模型＋地面、PhysicsScene、燈光、相機。左右輪已預填相當於 0.2 m/s 直行的目標輪速；按 Play 後用來測試物理移動。 |
| `AMR_D200_navigation.usd` | **簡化支撐版**。3 個剛體、2 個驅動關節；四組腳輪改用零摩擦球形支撐代理，腳輪外觀固定在車身上。可用於隔離腳輪動態問題，**不能當作完整腳輪力學模型**。 |
| `AMR_D200_navigation_demo.usd` | 簡化支撐版的測試場景。名稱中的 navigation 不代表已含 ROS 2／Nav2。 |
| `scripts/run_amr.py` | 使用 Isaac Sim 的 Python 環境啟動；含鍵盤、定速、動作示範與實際移動檢查模式。 |
| `scripts/teleop_in_isaac.py` | 在已開啟 Isaac Sim 的 Script Editor 使用鍵盤控制，不替換場景。 |
| `amr_parameters.json` | 質量、慣量估算依據、摩擦、驅動力與控制限制；幾何與物理參數修改後需重建 USD。 |
| `reports/` | 原始網格、圓柱擬合、碰撞簡化、關節連接及離線驗證報告。 |
| `source/amr_d200.stl` | 原始檔，未改動。 |

四個 USD 都是自包含文字編碼 USD（檔頭 `#usda 1.0`），沒有外部 USD 引用、貼圖或 Nucleus 依賴。即使只取出單一 demo USD，也包含模型及測試場景；鍵盤與自動測試需要完整解壓資料夾。

## 3. 最快的 GUI 測試

在 Isaac Sim 使用 **File → Open** 開啟 `AMR_D200_drive_demo.usd`。可切換到 `/World/OverviewCamera` 查看整車，然後按 **Play**。檔內已設定正向輪速，因此本步不需要 ROS 2 或先執行 Python 控制器。這是「已配置的測試」，不是交付環境已完成的動態驗證。

此純 USD 場景會持續給出直行命令，**不會自動避障或計時停止**；用 Pause／Stop 停止測試。它的地面大小為 40 × 40 m。主模型 `AMR_D200.usd` 本身的輪速則為零，不會一載入就自行行駛。

輪胎碰撞是 `Cylinder` prim。實際 PhysX 圓柱接觸實作受版本及設定影響：隨附腳本會設定 `/physics/collisionCylinderCustomGeometry = true`，以及新版對應的 `/physics/collisionApproximateCylinders = false`。單靠打開 USD 不會自動寫入這些全域偏好；若直接開檔後有接觸問題，先用下面的啟動程式測試。場景選用 CPU 物理；未驗證 GPU 物理模式。

## 4. Ubuntu：鍵盤控制

先完整解壓本包。以下假設解壓位置是 `~/Downloads/AMR_D200_IsaacSim`；請修改成你的實際路徑。`python.sh` 必須是 **Isaac Sim 安裝目錄內的啟動器**，不是系統 Python。

```bash
cd /你的/IsaacSim/安裝目錄
./python.sh "$HOME/Downloads/AMR_D200_IsaacSim/scripts/run_amr.py" --mode keyboard
```

啟動後會載入場景並開始模擬。點選視窗中的 viewport，使其收到鍵盤輸入。**W／S** 前進與後退，**A／D** 左轉與右轉；方向鍵也可用。**放開按鍵**會依設定減速，**Space** 把輪速命令歸零，**Esc** 結束控制。按鍵可能同時觸發 viewport 導航快捷鍵，但不應影響輪速命令；可保持 OverviewCamera 視角。

預設鍵盤前進命令為 0.30 m/s、轉向命令為 0.70 rad/s；它們是命令值，不是硬體實測速度。Space 不是安全認證急停，也不保證物理車身瞬間零速度。

使用簡化腳輪支撐版：

```bash
./python.sh "$HOME/Downloads/AMR_D200_IsaacSim/scripts/run_amr.py" --mode keyboard --navigation
```

本腳本需要提供 `isaacsim.SimulationApp` 的 Isaac Sim Python 環境。各版本相容性尚未在交付環境驗證；若使用 pip 安裝而沒有 `python.sh`，需用該 Isaac Sim Python 環境執行，不能直接用一般系統 Python。

### 已經開啟 Isaac Sim，不想再開一個視窗

先開啟 demo USD，保持 **Stop**，在 Script Editor 執行以下內容。只修改第一行檔案路徑：

```python
p = "/home/你的帳號/Downloads/AMR_D200_IsaacSim/scripts/teleop_in_isaac.py"
with open(p, encoding="utf-8") as f:
    exec(compile(f.read(), p, "exec"), {"__file__": p})
```

再點選 viewport、按 Play。腳本將 demo 預設直行命令改為零，等待鍵盤輸入。不要同時執行兩個控制程式或另外讓 OmniGraph／ROS 2 寫入相同關節；重複執行這支腳本會關閉它自己前一次的控制器。

## 5. 實際在你的 Isaac Sim 執行驗證

這裡**沒有執行 OpenUSD 回讀、PhysX 動態或 Isaac Sim UI 測試**。以下程式會在你電腦上產生真實測試紀錄；未執行前不能視為通過。

只做 OpenUSD 載入、API 與關節／網格資料檢查：

```bash
./python.sh "$HOME/Downloads/AMR_D200_IsaacSim/scripts/run_amr.py" --mode validate --headless
```

執行約 12 秒的模擬時間測試，依序靜置、前進、停車、原地旋轉、後退、停止：

```bash
./python.sh "$HOME/Downloads/AMR_D200_IsaacSim/scripts/run_amr.py" --mode check --headless
```

結果會寫到 `reports/isaac_runtime_result.json`；每次執行覆寫該檔，可用 `--output /完整路徑/result.json` 指定其他檔案。`PASS_PHYSICS_SMOKE_TEST` 才表示此測試序列的實際位移、轉角與姿態檢查均通過。`PASS_OPENUSD_CHECK` **只代表資料檢查，不代表動態測試通過**。

動態基本門檻：前進段正向位移 > 0.25 m；旋轉段正向轉角 > 0.45 rad；後退段位移 < −0.10 m；車身參考點離地高度在 0.03～0.15 m；車身向上向量 Z 分量 > 0.95。這些是冒煙測試門檻，不是運動精度、承載或工業安全驗收。

可用 `--mode demo` 顯示同一組動作而不執行門檻驗收。定速範例：

```bash
./python.sh "$HOME/Downloads/AMR_D200_IsaacSim/scripts/run_amr.py" --mode constant --linear 0.2 --angular 0 --seconds 10
```

## 6. 加入既有場景／接上自己的控制器

單位為 **m、kg、s**；**+X 前進、+Y 左側、+Z 向上**。來源座標轉換為 `X_new = Y_STL, Y_new = -X_STL, Z_new = Z_STL`，不是鏡射。資產根節點原點位於標稱地面；`base_link` 原點位於左右驅動輪軸線高度。

將 `AMR_D200.usd` 引用到 `/World/AMR_D200`，整個根節點可先設 Z=0.003 m，避免初始穿入地板。既有場景須具備適當的 PhysicsScene、重力和地面碰撞。**不要額外加世界固定關節、不要把 base_link 設為 kinematic，也不要縮放剛體來修改物理尺寸。**

| 參數／路徑 | 值 |
|---|---|
| 資產根節點 | `/AMR_D200`；demo 中為 `/World/AMR_D200` |
| Articulation Root | 資產根節點下的 `base_link`，不是單純的最上層容器 |
| 左驅動關節 | `joints/left_wheel_joint` |
| 右驅動關節 | `joints/right_wheel_joint` |
| 輪半徑 r | 0.0762 m |
| 左右驅動輪中心距 b | 0.586 m |
| 兩個驅動關節正軸 | 都為局部 +Y；兩輪正轉對應車身 +X 前進 |
| LiDAR 安裝節點 | `base_link/SensorMounts/lidar_mount` |
| 載荷安裝節點 | `base_link/payload_mount` |

以關節**名稱**選取左右輪，不要假設它們一定是某個 API 回傳陣列中的第 0、1 項；完整版同時包含 12 個被動自由度。差速輪速公式為：

```text
left_rad_s  = (linear_m_s - angular_rad_s * track_m / 2) / radius_m
right_rad_s = (linear_m_s + angular_rad_s * track_m / 2) / radius_m
```

`D200Drive.set_twist(v, yaw_rate, dt)` 接收 m/s 與 rad/s，內部會轉換。**USD 的 angular DriveAPI `targetVelocity` 單位是 deg/s**；不要把 rad/s 原樣寫入。若改接 Isaac 的 articulation 輪速控制 API，請依該 API 的單位使用 rad/s，不要再次套用 USD 的角度轉換。原始 drive 設定為 force、stiffness=0，以 damping 與 maxForce 限制速度驅動；沒有位置鎖定。

## 7. 幾何、碰撞與近似範圍

原始 STL 為 **8,260 三角面／59 個連通元件**，59 個元件的封閉性與面片繞向檢查全部通過。輪胎、輪轂、螺栓和其他可辨識圓柱共 **44 個元件**改為具有 radius、height、axis 的 USD `Cylinder`；10 個方塊改為 `Cube`；5 個車身網格保留原始頂點與面。這是 USD 幾何整理，不是 CAD 特徵樹重建。

圓柱是從原始 32 邊輪廓擬合。最大來源頂點徑向殘差約 **0.0000141 mm**，但真圓與來源多邊形弦段之間最大理論弓高約 **0.366924 mm**。前一數字不代表全表面誤差，也不代表實物製造精度。本次沒有把「擬合到頂點」誤稱為零形狀變化。

車身碰撞用 5 個凸包代理，每個 104 頂點；其擠出輪廓經離線檢查與來源投影外輪廓一致，但不是車身完整凹槽、孔洞或倒角的精確碰撞。輪胎則用獨立圓柱碰撞，不直接讓 STL 小平面當滾動接觸面。外觀元件存在互相重疊；為避免內部互相撞擊，本版關閉 articulation self-collision。未做整機自交、配合間隙、製造或承載驗收。

完整版：**15 剛體／14 關節／19 碰撞形狀**，其中 2 個驅動輪關節、4 個被動腳輪轉向關節、8 個被動腳輪滾動關節。腳輪保留約 15 mm 的拖距，雙輪可各自滾動。輪胎沒有變形模型；車架與腳輪沒有彈簧懸吊。

## 8. 物理參數不是原廠規格

| 設定 | 本次模擬初始值 |
|---|---|
| 全車總質量 | **80 kg（假設）**；base 72、驅動輪各 2、腳輪支架各 0.5、八顆腳輪胎各 0.25 kg |
| 車身重心 | 假設標稱地面上方 0.14 m；慣量用簡化幾何估算 |
| 每側最大驅動力矩 | **15 N·m（假設）** |
| 輪胎靜／動摩擦 | **0.9／0.8（假設）** |
| 控制限速 | 0.5 m/s、1.0 rad/s；不是硬體能力評估 |
| 碰撞參數 | contactOffset=1 mm；腳輪 restOffset=−0.2 mm，驅動輪為 0 |
| 場景物理 | CPU、TGS、120 Hz；不是性能或穩定性保證 |

沒有把檔名 D200 解讀成自重或載重 200 kg。載荷安裝節點不會自動增加質量；加上其他模組時需重算質量、重心、慣量與接觸負載。摩擦、扭矩、慣量、腳輪拖曳及輪速追蹤都需在實際 Isaac Sim 與你的用途下校準。

**LiDAR 目前只有外觀與安裝座標，沒有啟用感測器。未配置 ROS 2 topics、里程計、TF、SLAM、Nav2、路徑規劃或避障。** 本包處理的是可驅動底盤的模型與測試入口，而不是完整自主導航系統。

## 9. 驗證層級與重建

交付環境已完成：原始網格分件與封閉性、原始檔 SHA-256、圓柱擬合／輪心與軸線、所有59元件完整性、質量與慣量數值、關節樹與錨點、所有內部路徑引用、本文字 USD 子集的獨立 Lark 解析、純 Python 差速計算測試及 Python 語法編譯。

**離線檢查器不是 Pixar OpenUSD parser，也不是 NVIDIA asset validator。尚未完成：真正 OpenUSD 開檔、PhysX 碰撞 cooking、實際前進／轉向、GUI 顯示、API 版本相容性、CPU／GPU 效能及長時間穩定性。** `reports/offline_validation.json` 也明確記錄這兩個 runtime-tested 旗標為 false。

需要修改質量／摩擦／馬達設定時，編輯 `amr_parameters.json` 後，在獨立的建檔用 Python 環境執行：

```bash
python scripts/build_asset.py
python scripts/check_delivery_offline.py
python scripts/drive_math.py
```

建檔用依賴在 `requirements-build.txt`，不是開啟 USD 或執行控制所需的依賴。不要為了建檔任意改動 Isaac Sim 內建的 numpy／scipy 環境。JSON 中的 `command_limits` 會由控制程式啟動時讀取；其他物理參數需要重建才會影響 USD。重建或編輯後，原始 `SHA256SUMS.txt` 不再適用。

## 10. 校驗下載完整性

在本資料夾執行：

```bash
sha256sum -c SHA256SUMS.txt
```

這只校驗交付位元組是否一致，不代表動態驗證通過。ZIP 交付檢查另見壓縮包外的完整性紀錄。

## 技術依據（官方文件；不代表本資產已由 NVIDIA 驗證）

- USD DriveAPI：角速度目標單位、force drive、stiffness／damping／maxForce。https://docs.omniverse.nvidia.com/kit/docs/usdrt.scenegraph/7.5.0/api/classusdrt_1_1_usd_physics_drive_a_p_i.html
- Isaac Sim differential controller：左右輪的差速運動學。https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_simulation/mobile_robot_controllers.html
- Omni Physics settings：USD transform updates、圓柱碰撞相關設定。https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.2/dev_guide/settings.html
- PhysX Cylinder 支援與近似設定。https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/population/Cylinder.html

上述連結是實作參照，**不是對任何特定 Isaac Sim 版本的測試認證**。
