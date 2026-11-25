# Paper trading and daily automation (Phase 6)

Run a strategy forward on new data before trusting it with money. There are two kinds of
portfolio:

- **paper**: simulated fills at the next day's open, with real charges and circuit checks.
- **manual**: the system tells you what to trade, you place the orders at Zerodha (or any
  broker), then record your fills. Use this for your real money.

## Start
```bash
alphaengine db upgrade
alphaengine portfolio create lowvol_paper --strategy low_vol_momentum --capital 1000000
alphaengine portfolio create core --strategy fixed_mix --capital 500000 --broker manual --band 0.03
alphaengine scheduler run-eod            # once, to see it work (downloads today's data)
alphaengine portfolio show lowvol_paper
```
A portfolio rebalances on its first day and then on its strategy's schedule: the last trading
day of the month or quarter, with signals at the close and trades the next morning.

## Every day (automatic)
`alphaengine scheduler run-eod` (idempotent per date):
1. Downloads the day's bhavcopy. If NSE hasn't published it yet, the run is marked `waiting`.
2. For each active portfolio: applies dividends, splits and bonuses, fills paper orders at
   today's open, marks to market, and alerts if the portfolio is 15% or more below its peak.
3. On a rebalance day: computes targets and the trade list, runs risk checks, and creates orders.
4. Sends one summary to Telegram, or to `var/data/notifications.log` if Telegram isn't set up.

**Windows Task Scheduler** (run from the repo folder in PowerShell). `run_eod_hidden.vbs` runs
the job without a console window. Closing a visible window would kill the job.
```powershell
$tr = "wscript.exe `"$PWD\scripts\run_eod_hidden.vbs`""
schtasks /Create /TN "AlphaEngine EOD" /TR $tr /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 19:45 /F
schtasks /Create /TN "AlphaEngine EOD retry" /TR $tr /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 21:30 /F
$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Set-ScheduledTask -TaskName "AlphaEngine EOD" -Settings $s; Set-ScheduledTask -TaskName "AlphaEngine EOD retry" -Settings $s
```
The retry is a no-op once the day succeeded. With StartWhenAvailable, a run missed while the PC
was off happens as soon as it's back on. Output goes to `var\eod.log`. Tasks run the code that is
**checked out in this folder**, so keep a branch with all phases checked out. You can also keep
`alphaengine scheduler start` running instead; it retries every 30 minutes until 23:30.

**Telegram**: create a bot with @BotFather, send it a message, get your chat id from
`https://api.telegram.org/bot<TOKEN>/getUpdates`, then add to `.env`:
```
ALPHAENGINE_TELEGRAM_BOT_TOKEN=...
ALPHAENGINE_TELEGRAM_CHAT_ID=...
```
Check it with `alphaengine notify test`.

## Manual portfolio routine (rebalance days only, ~10 minutes)
```bash
alphaengine portfolio checklist core --out orders.csv   # place SELLs first, then BUYs
alphaengine portfolio fill 12 --qty 974 --price 102.40 --charges 18.6   # per order, from the contract note
alphaengine portfolio reconcile core holdings.csv        # Zerodha Console holdings export
```
Already own shares? Load them as lots with their real purchase dates (this matters for tax):
`alphaengine portfolio import-holdings core lots.csv`, where the CSV has
`symbol,quantity,avg_price,acquired`.

## Tax-aware planning
When a sell would realise short-term gains on lots that turn long-term within 30 days, the
trade shows the estimated tax saved by waiting. **Trims** are deferred automatically, while
**exits** still follow the strategy but carry the note. `alphaengine portfolio tax <name>`
estimates capital-gains tax per FY from realised sales.

## Safety
- `alphaengine portfolio pause <name>` is the kill switch: no new orders until `resume`.
- Risk checks block plans that put more than 25% in one stock (index ETFs are exempt) or trade
  more than 150% of the portfolio, and warn when an order is large compared with daily volume.
- Orders are idempotent per (portfolio, signal date, symbol, side), so re-running a day never
  duplicates them.
- Nothing is ever sent to a broker. Automatic order placement (Phase 9) will stay behind explicit
  approval.
