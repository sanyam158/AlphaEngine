import { NavLink, Route, Routes } from "react-router-dom";
import BacktestDetailPage from "./pages/BacktestDetail";
import BacktestsPage from "./pages/Backtests";
import DataPage from "./pages/Data";
import GoalPage from "./pages/Goal";
import HomePage from "./pages/Home";
import JournalPage from "./pages/Journal";
import PortfolioPage from "./pages/Portfolio";
import StrategiesPage from "./pages/Strategies";

const NAV = [
  { to: "/", label: "Home", end: true },
  { to: "/backtests", label: "Backtest lab" },
  { to: "/strategies", label: "Strategies" },
  { to: "/goal", label: "Goal planner" },
  { to: "/journal", label: "Journal" },
  { to: "/data", label: "Data health" },
];

export default function App() {
  return (
    <div className="shell">
      <nav className="nav" aria-label="Main">
        <div className="brand">
          <img src="/favicon.svg" alt="" />
          AlphaEngine
        </div>
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to} end={n.end}>
            {n.label}
          </NavLink>
        ))}
        <div className="foot">Research & paper trading. Not investment advice.</div>
      </nav>
      <main>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/portfolios/:name" element={<PortfolioPage />} />
          <Route path="/backtests" element={<BacktestsPage />} />
          <Route path="/backtests/:runId" element={<BacktestDetailPage />} />
          <Route path="/strategies" element={<StrategiesPage />} />
          <Route path="/journal" element={<JournalPage />} />
          <Route path="/goal" element={<GoalPage />} />
          <Route path="/data" element={<DataPage />} />
          <Route path="*" element={<p>Not found.</p>} />
        </Routes>
      </main>
    </div>
  );
}
