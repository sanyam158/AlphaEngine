"""Exception hierarchy. Catch ``AlphaEngineError`` to handle anything raised by the system."""


class AlphaEngineError(Exception):
    """Base class for all AlphaEngine errors."""


class ConfigError(AlphaEngineError):
    """Invalid or missing configuration."""


class DataError(AlphaEngineError):
    """Missing, malformed or inconsistent market data."""


class DataSourceError(DataError):
    """An external data source failed (network, rate limit, bad payload)."""


class LookAheadError(AlphaEngineError):
    """A computation tried to use information not available at its as-of time."""


class StrategyError(AlphaEngineError):
    """A strategy is misconfigured or produced an invalid target portfolio."""


class OrderError(AlphaEngineError):
    """An order is invalid or an illegal order-state transition was attempted."""


class RiskLimitError(AlphaEngineError):
    """A pre-trade or portfolio risk check failed."""
