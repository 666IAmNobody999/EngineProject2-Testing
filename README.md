# 幻神揭棋引擎（PikaJieQi）

以 PikaJieQi（官方 Pikafish `jieqi_old` 分支）為基礎的揭棋引擎，並移植了幻神引擎
（[666IAmNobody999/EngineProject-Testing](https://github.com/666IAmNobody999/EngineProject-Testing)）
中可通用的部分。引擎使用傳統評估（`USE_NNUEEVAL 0`），不需要 NNUE 網絡檔。

## 建置

```bash
./build_engine.sh                 # x86-64-bmi2 + PGO，輸出 pikajieqi-native
ARCH=x86-64-avx2 ./build_engine.sh
cd src && make -j profile-build ARCH=x86-64-bmi2 COMP=mingw   # Windows (msys2)
```

GitHub Actions（`.github/workflows/pikajieqi.yml`）會為 Windows／Linux 各建置
`x86-64-bmi2`、`x86-64-avx2`、`x86-64-sse41-popcnt` 三款 PGO 版本並上載為 artifact。

## 從幻神引擎移植的項目

| 項目 | 說明 |
|---|---|
| PGO 建置流程 | `build_engine.sh` 及 GitHub Actions 均改用 `profile-build`，實測比原本的 sse41 無 PGO 建置快約 7% |
| seekMate 殺棋模式 | 根節點分數達 `750 + 220000/depth²` 時：關閉 razoring／小型 ProbCut／單步延伸、空著縮減減半、所有將軍著法延伸、LMR 對將軍著法減少縮減 |
| Windows 效能設定（`winperf.cpp`） | 處理程序 CPU／IO 高優先級、啟動「幻神終極效能」電源計劃並於退出時還原、把其他 pikafish／pikajieqi 處理程序降為閒置優先級、搜尋執行緒逐一綁定實體核心 |
| 引擎名稱 | `幻神揭棋引擎 YYYYMMDD` |
| 預設 Hash | 16 MB → 256 MB（揭子節點非常依賴置換表） |

## 揭子（機會節點）搜尋的改動

揭棋的每一步暗子著法都是機會節點：引擎會把暗子展開成所有可能的棋子種類，
分別搜尋後以勝率平均合併。這一版的改動：

1. **DarkMode 1（預設）**：超出揭子預算（`DarkMaxDepth`，預設 2 層）後，暗子不再展開，
   而是維持「未知棋子」（以剩餘暗子的平均價值估算）繼續全深度搜尋。原本的做法是把所有結果
   展開但深度歸零，令開局的有效深度只有 2–3 層（報告 depth 17 但 seldepth 3）；
   改動後 3 秒搜尋可達 depth 18–26、seldepth 17–22。
2. **Star1 式剪枝**（`DarkPrune`）：每搜完一個揭子結果就計算合併值的上下界，
   當餘下結果無論如何都無法超過 alpha（或已必然超過 beta）時提早結束展開。
3. **殺棋分數修正**：合併前不再把分數截斷到 ±2862，揭子後的殺著能正確保留殺棋距離。
4. **假殺修正**：揭子預算耗盡時不再直接跳過著法（被將軍時會誤判為被將死），
   改為保留暗子繼續搜尋。
5. **速度**：移除每步都建立的除錯字串，`ScoreCalc` 改用固定陣列並以增量方式維護勝率總和。

所有參數都可用 UCI 選項調整（亦可用於 SPSA）：
`DarkMode`、`DarkMaxDepth`、`DarkQDepth`、`DarkMaxTypes`、`DarkFlipScaling`（K 值，預設 361）、
`DarkIndependent`、`DarkPrune`、`SeekMateOn`。

## 自對弈測試

`tools/jieqi_match.py` 是一個簡單的揭棋裁判程式：由裁判持有暗子袋、隨機決定翻出的棋子，
並以引擎的著法格式（`b2e2C`、`e3e6Pn`）把同一著法序列送給兩個引擎。

```bash
python3 tools/jieqi_match.py --e1 ./A --e2 ./B --games 100 --movetime 100 \
    --o1 "Hash=32" --o2 "Hash=32,DarkMode=0" --concurrency 2
```

本版測試結果（每場 100 局、每步 100 ms、2 核心）：

| 對比 | 結果 |
|---|---|
| 新版（全部改動）vs 原版 | +58 −37 =5，約 +74 ± 69 Elo |
| DarkMode 1 vs DarkMode 0（其餘相同） | +57 −40 =3，約 +60 ± 69 Elo |
| DarkMaxDepth 1 / 3 vs 2 | 皆在誤差範圍內 |
| 每個揭子結果獨立決定重搜 vs 沿用最小值 | 獨立重搜較差（約 −75 Elo），故預設 `DarkIndependent 0` |

樣本仍少，較長時限及更多局數的驗證仍有需要。
