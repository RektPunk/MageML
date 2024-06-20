from typing import Dict
import pandas as pd


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 모델에 사용할 컬럼, 컬럼의 rename rule을 미리 할당함
    cols_dict: Dict[str, str] = {
        "ID": "ID",
        "target": "target",
        "_type": "_type",
        "hourly_market-data_coinbase-premium-index_coinbase_premium_gap": "coinbase_premium_gap",
        "hourly_market-data_coinbase-premium-index_coinbase_premium_index": "coinbase_premium_index",
        "hourly_market-data_funding-rates_all_exchange_funding_rates": "funding_rates",
        "hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations": "long_liquidations",
        "hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations_usd": "long_liquidations_usd",
        "hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations": "short_liquidations",
        "hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations_usd": "short_liquidations_usd",
        "hourly_market-data_open-interest_all_exchange_all_symbol_open_interest": "open_interest",
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_ratio": "buy_ratio",
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_sell_ratio": "buy_sell_ratio",
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume": "buy_volume",
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_ratio": "sell_ratio",
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume": "sell_volume",
        "hourly_network-data_addresses-count_addresses_count_active": "active_count",
        "hourly_network-data_addresses-count_addresses_count_receiver": "receiver_count",
        "hourly_network-data_addresses-count_addresses_count_sender": "sender_count",
    }
    df = df[cols_dict.keys()].rename(cols_dict, axis=1)
    return df
