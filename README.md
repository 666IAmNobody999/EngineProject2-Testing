# PikaJieQi Enhanced — Cờ Úp Bot

Bot cờ úp (mystery_xiangqi) cho gamevh.net với engine **PikaJieQi Enhanced**.

## Thuật toán đã port

| # | Algorithm | Source | Impact |
|---|---|---|---|
| 1 | Sigmoid flip aggregation | Official Pikafish jieqi (9b963f7) | +75% win rate |
| 2 | Chance branch reduction | AB-JChess | Giảm depth cho pawn/advisor reveals |
| 3 | Mate fix in flip_search | Official Pikafish jieqi (10334fe) | Correct mate detection |

## Cấu trúc

```
cup_bot_jieqi.py              # Bot chính (Python, bet=1000)
build_engine.sh               # Build PikaJieQi + download NNUE
src/                          # Engine source (C++)
.github/workflows/
  test_engine.yml             # Build + test engine
  run_bot.yml                 # Chạy bot trên gamevh.net (mỗi 6h)
```

## Chạy local

```bash
# 1. Build engine
bash build_engine.sh

# 2. Install deps
pip install websocket-client requests

# 3. Run bot (bet=1000 xu)
python3 cup_bot_jieqi.py
```

## Chạy trên GitHub Actions

Workflow `run_bot.yml` chạy mỗi 6 giờ:
1. Build PikaJieQi Enhanced từ source
2. Download NNUE
3. Run bot (timeout 340 phút, bet=1000 xu)

### Setup Secrets

Vào **Settings → Secrets → Actions**:

| Secret | Value |
|---|---|
| `CARO_USER19` | tài khoản gamevh.net |
| `CARO_PASSWD19` | mật khẩu |

Nếu không set, bot dùng default (`nguyen15` / `nhat123456`).

## Engine

- **Base**: `brianhliou/pikafish-jieqi-wasm` (jieqi_old-mistboard, commit e75cee3a)
- **NNUE**: Standard `pikafish.nnue` từ `official-pikafish/Networks`
- **Depth**: 19-25 trong 5s search
- **go infinite + stop** pattern (go movetime không hoạt động trong jieqi branch)

## License

GPL v3 (inherited from Pikafish)
