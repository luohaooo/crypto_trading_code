# Market-Neutral Backtest Framework

Please create a new folder in `@backtest` to implement a *market-neutral* strategy.  

The strategy works as follows:  
- In each time window, allocate **half of the capital** equally across the **top-n symbols** (by factor value) for **long** positions.  
- Allocate the **other half** equally across the **bottom-n symbols** for **short** positions.  
- A *factor* here means an extracted data feature — for example, the return over the past four hours can be used as a factor.  
- At the end of each time window, close the previous round of positions and open new positions for the next window according to the factor values.  
- This process is called **rebalancing**.  

---

## Example Flow

1. On the interface, the user chooses the rebalancing frequency and optional factors.  
2. The backend fetches data for all symbols and computes factor values for every symbol.  
3. The user selects `n`, transaction fees, and the backtest period on the interface and clicks **Start Backtest**.  
   - The backend computes returns and records trade information.  
   - The P&L curve is shown on the interface in real time.  
   - Trade information is displayed in a table that records, for each time window, which symbols were traded and what the returns were.  
4. When the backtest finishes:  
   - Compute and display performance metrics (e.g., total return, Sharpe ratio, etc.).  
   - Provide a **Download** button so the user can download the backtest report (including the P&L chart, performance metrics, and trade log).  
5. The user can choose to store computed factors so that future backtests can skip factor computation.  

---

## Requirements

1. Proper memory management is required to **prevent out-of-memory (OOM) issues**.  
2. The framework must ensure **fast factor computation** and **high backtesting speed**.  

Use two parallel subagents to brainstorm possible plans. Do not implement any code.