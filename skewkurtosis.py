import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from jquants_derivatives import Client, Option, database


class SkewKurtosis:
    def __init__(self, date: str, target_delta: float = 0.25, days: float = 30):
        self.days = days
        self.target_delta = target_delta
        self.data = Option(Client().get_option_index_option(date), contracts=2)
        near = self.data.contract_month[0]
        far = self.data.contract_month[1]
        self.iv_ = (
            pd.merge(
                self.data.contracts_dfs[near].loc[
                    :, ["StrikePrice", "ImpliedVolatility"]
                ],
                self.data.contracts_dfs[far].loc[
                    :, ["StrikePrice", "ImpliedVolatility"]
                ],
                on="StrikePrice",
                how="left",
                suffixes=["_near", "_far"],
            )
            .set_index("StrikePrice")
            .sort_index()
        )
        self.delta_ = (
            pd.merge(
                self.data.contracts_dfs[near].loc[:, ["StrikePrice", "Delta"]],
                self.data.contracts_dfs[far].loc[:, ["StrikePrice", "Delta"]],
                on="StrikePrice",
                how="left",
                suffixes=["_near", "_far"],
            )
            .set_index("StrikePrice")
            .sort_index()
        )
        self.t = days / 365
        if (np.array(list(self.data.time_to_maturity.values())) < self.t).all():
            # 期近が基準日未満の場合は期近のデータを使う
            self.s = self.data.underlying_price[
                sorted(self.data.underlying_price.keys())[0]
            ]
            self.r = self.data.interest_rate[sorted(self.data.interest_rate.keys())[0]]
            self.iv = self.iv_.loc[:, "ImpliedVolatility_near"]
            self.delta_.loc[:, "Delta_near"]
        else:
            xp = sorted(self.data.time_to_maturity.values())
            self.s = np.interp(self.t, xp, sorted(self.data.underlying_price.values()))
            self.r = np.interp(self.t, xp, sorted(self.data.interest_rate.values()))
            self.iv = self.iv_.apply(lambda fp: np.interp(self.t, xp, fp), axis=1)
            self.delta = self.delta_.apply(lambda fp: np.interp(self.t, xp, fp), axis=1)
        self.base_volatility = self.get_interp_iv(self.iv, self.s)
        self.skew = self.get_skew(self.target_delta)
        self.kurtosis = self.get_kurtosis()

    def get_interp_strike(self, ser: pd.Series, target: float):
        nearest_ix = abs(ser - target).idxmin()
        next_nearest_ix = abs(ser.drop(nearest_ix) - target).idxmin()
        if nearest_ix < next_nearest_ix:
            near_ser = ser.loc[nearest_ix:next_nearest_ix].sort_values()
        else:
            near_ser = ser.loc[next_nearest_ix:nearest_ix].sort_values()
        return np.interp(target, near_ser, near_ser.index)

    def get_interp_iv(self, ser: pd.Series, target: float):
        interp_ser = ser.copy()
        interp_ser.loc[target] = np.nan
        return interp_ser.sort_index().interpolate().loc[target]

    def get_skew(self, target: float | None = None):
        if not target:
            target = self.target_delta
        k_lower = self.get_interp_strike(self.delta, -target)
        k_upper = self.get_interp_strike(self.delta, target)
        iv_lower = self.get_interp_iv(self.iv, k_lower)
        iv_upper = self.get_interp_iv(self.iv, k_upper)
        skew = iv_lower - iv_upper
        return {"strike": (k_lower, k_upper), "iv": (iv_lower, iv_upper), "skew": skew}

    def get_kurtosis(self, target: float | None = None):
        if not target:
            target = self.target_delta
        otm = self.get_interp_iv(
            pd.DataFrame(self.get_skew(target))
            .set_index("strike")
            .loc[:, "iv"]
            .sort_values(),
            self.s,
        )
        atm = self.base_volatility
        kurtosis = otm - atm
        return {"otm": otm, "atm": atm, "kurtosis": kurtosis}

    def plot_skew_kurt(self):
        shape_x = list(self.skew["strike"])
        shape_x.append(self.s)
        shape_y = list(self.skew["iv"])
        shape_y.append(self.get_interp_iv(self.iv, self.s))
        kurt_x = [self.s, self.s]
        kurt_y = [self.kurtosis["atm"], self.kurtosis["otm"]]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.iv_.index,
                y=self.iv_.loc[:, "ImpliedVolatility_near"],
                name="期近",
                line={"width": 0.5},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.iv_.index,
                y=self.iv_.loc[:, "ImpliedVolatility_far"],
                name="期先",
                line={"width": 0.5},
            )
        )
        fig.add_trace(go.Scatter(x=self.iv.index, y=self.iv, name=f"{self.days}日補間"))
        fig.add_trace(go.Scatter(x=self.skew["strike"], y=self.skew["iv"], name="skew"))
        fig.add_trace(go.Scatter(x=kurt_x, y=kurt_y, name="kurtosis"))
        fig.add_trace(
            go.Scatter(x=shape_x, y=shape_y, fill="toself", mode="lines", opacity=0.5)
        )
        fig.show()
