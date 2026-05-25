import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from numba import njit

from hftbacktest import (
    ALL_ASSETS,
    BUY_EVENT,
    SELL_EVENT,
    BacktestAsset,
    GTX,
    LIMIT,
    HashMapMarketDepthBacktest,
    Recorder,
)
from hftbacktest.order import IOC, MARKET

from bitmex_single_market_mm_backtest import (
    BITMEX_CONTRACT_SIZE,
    BITMEX_EXCHANGE,
    BITMEX_LOT_SIZE,
    BITMEX_ORDER_ENTRY_LATENCY_NS,
    BITMEX_ORDER_QTY,
    BITMEX_ORDER_RESPONSE_LATENCY_NS,
    BITMEX_SYMBOL,
    BITMEX_TICK_SIZE,
    CSV_DIR,
    NPZ_DIR,
    RESULT_DIR,
    convert_bitmex,
    download_file,
    end_close_ts_ns,
    tardis_key,
)


ORDER_UPDATE_INTERVAL_NS = 10_000_000
ORDER_INFLIGHT_NS = 80_000_000
REST_MIN_INTERVAL_NS = 700_000_000
ORDER_TTL_NS = 5_000_000_000
MIN_AMEND_TICKS = 5.0

SIGNAL_HISTORY_LEN = 8192
FLOW_DECAY = 0.92

ORDER_QTY = BITMEX_ORDER_QTY
BASE_HALF_SPREAD_BPS = 3.0
MIN_HALF_SPREAD_TICKS = 1.0
VOL_SPREAD_MULTIPLIER = 0.3
INVENTORY_SKEW_BPS_AT_SOFT_LIMIT = 4.0
INVENTORY_SPREAD_BPS_AT_SOFT_LIMIT = 1.5
SOFT_POSITION_CONTRACTS = 500.0
MAX_POSITION_CONTRACTS = 1_000.0
USE_DYNAMIC_QUOTE = False
DYNAMIC_QUOTE_EXPECTED_EDGE_MULT = 1.0
DYNAMIC_QUOTE_EDGE_IF_FILLED_MULT = 0.25
DYNAMIC_QUOTE_MAX_TIGHTEN_BPS = 2.0
DYNAMIC_QUOTE_FILL_PROB_WIDEN_MULT = 0.0
DYNAMIC_QUOTE_FILL_PROB_BASELINE = 0.20
DYNAMIC_QUOTE_MAX_WIDEN_BPS = 2.0

MAKER_FEE_RATE = 0.0
TAKER_FEE_RATE = 0.0001
EXCHANGE_MODEL = "no_partial"
RESULT_TAG = ""
MIN_EXPECTED_EDGE_BPS = 0.02
MIN_EDGE_IF_FILLED_BPS = -999.0
MIN_FILL_PROB = 0.02
MIN_PLACEMENT_EXPECTED_MARGIN_BPS = 0.0
MIN_PLACEMENT_EDGE_IF_FILLED_MARGIN_BPS = 0.0
USE_INTRADAY_PERCENTILE_GATE = False
EXPECTED_EDGE_PERCENTILE = 0.0
EDGE_IF_FILLED_PERCENTILE = 0.0
FILL_PROB_PERCENTILE = 0.0
REDUCE_ONLY_AFTER_SOFT_POSITION = False
DAILY_LOSS_LIMIT_USDT = 0.0
DAILY_FILL_LIMIT = 0
USE_REGIME_EXPECTED_EDGE_GATE = False
REGIME_EXPECTED_EDGE_WARMUP_SAMPLES = 6_000
REGIME_EXPECTED_EDGE_EWM_ALPHA = 0.0001
REGIME_MIN_BID_EXPECTED_EDGE_BPS = -0.02
REGIME_MIN_ASK_EXPECTED_EDGE_BPS = -0.02
PERCENTILE_WARMUP_SAMPLES = 6_000
PERCENTILE_UPDATE_INTERVAL_NS = 1_000_000_000
PERCENTILE_BINS = 512
EXPECTED_EDGE_HIST_MIN = -0.5
EXPECTED_EDGE_HIST_MAX = 0.5
EDGE_IF_FILLED_HIST_MIN = -2.0
EDGE_IF_FILLED_HIST_MAX = 2.0
FILL_PROB_HIST_MIN = 0.0
FILL_PROB_HIST_MAX = 0.35

FACTOR_IDS = {
    "spread_bps": 0,
    "queue_imbalance": 1,
    "microprice_bps": 2,
    "trade_flow_imbalance": 3,
    "trade_flow_ewm_imbalance": 4,
    "momentum_100ms_bps": 5,
    "momentum_250ms_bps": 6,
    "momentum_1000ms_bps": 7,
    "vol_1000ms_bps": 8,
    "depth_imbalance_3": 9,
    "depth_imbalance_5": 10,
    "weighted_depth_imbalance_5": 11,
    "bid_depth_slope_5": 12,
    "ask_depth_slope_5": 13,
    "top_bid_qty_change": 14,
    "top_ask_qty_change": 15,
    "ofi": 16,
    "ofi_1000ms": 17,
    "trade_qty_1000ms": 18,
    "trade_count_1000ms": 19,
    "momentum_3000ms_bps": 20,
    "vol_250ms_bps": 21,
}

MODEL_KEYS = (
    "bid_expected_edge_bps",
    "ask_expected_edge_bps",
    "bid_edge_if_filled_bps",
    "ask_edge_if_filled_bps",
    "bid_fill_prob",
    "ask_fill_prob",
)

FILL_ATTRIBUTION_FIELDS = [
    "date",
    "fill_index",
    "timestamp_ns",
    "side",
    "fill_count",
    "delta_contracts",
    "position_before",
    "position_after",
    "exec_px",
    "mid",
    "spread_capture_usdt_per_btc",
    "equity_usdt",
    "equity_delta_since_prev_fill_usdt",
    "bid_expected_edge_bps",
    "ask_expected_edge_bps",
    "bid_edge_if_filled_bps",
    "ask_edge_if_filled_bps",
    "bid_fill_prob",
    "ask_fill_prob",
    "side_expected_edge_bps",
    "side_edge_if_filled_bps",
    "side_fill_prob",
    "side_expected_threshold_bps",
    "side_edge_if_filled_threshold_bps",
    "side_fill_prob_threshold",
    "side_expected_margin_bps",
    "side_edge_if_filled_margin_bps",
    "side_fill_prob_margin",
    "bid_regime_expected_edge_ewm_bps",
    "ask_regime_expected_edge_ewm_bps",
    "placement_valid",
    "placement_action",
    "placement_timestamp_ns",
    "placement_age_ns",
    "placement_target_px",
    "placement_half_spread_bps",
    "placement_position_contracts",
    "placement_bid_expected_edge_bps",
    "placement_ask_expected_edge_bps",
    "placement_bid_edge_if_filled_bps",
    "placement_ask_edge_if_filled_bps",
    "placement_bid_fill_prob",
    "placement_ask_fill_prob",
    "placement_side_expected_edge_bps",
    "placement_side_edge_if_filled_bps",
    "placement_side_fill_prob",
    "placement_side_expected_threshold_bps",
    "placement_side_edge_if_filled_threshold_bps",
    "placement_side_fill_prob_threshold",
    "placement_side_expected_margin_bps",
    "placement_side_edge_if_filled_margin_bps",
    "placement_side_fill_prob_margin",
    "placement_bid_regime_expected_edge_ewm_bps",
    "placement_ask_regime_expected_edge_ewm_bps",
]

PLACEMENT_RECORD_LEN = 24


@njit
def ceil_to_tick(px, tick_size):
    return math.ceil(px / tick_size) * tick_size


@njit
def floor_to_tick(px, tick_size):
    return math.floor(px / tick_size) * tick_size


@njit
def clamp(value, low, high):
    return min(max(value, low), high)


@njit
def ratio_minus_one_bps(numerator, denominator):
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return 0.0
    return (numerator / denominator - 1.0) * 10_000.0


@njit
def bitmex_equity_usdt(hbt):
    depth = hbt.depth(0)
    mid = (depth.best_bid + depth.best_ask) / 2.0
    state = hbt.state_values(0)
    return state.balance + state.position * mid * BITMEX_CONTRACT_SIZE - state.fee


@njit
def cancel_all_orders(hbt):
    orders = hbt.orders(0)
    values = orders.values()
    while values.has_next():
        order = values.get()
        if order.cancellable:
            hbt.cancel(0, order.order_id, False)


@njit
def pair_ofi(bid, ask, bid_qty, ask_qty, prev_bid, prev_ask, prev_bid_qty, prev_ask_qty):
    bid_part = 0.0
    ask_part = 0.0
    if bid > prev_bid:
        bid_part = bid_qty
    elif bid == prev_bid:
        bid_part = bid_qty - prev_bid_qty
    else:
        bid_part = -prev_bid_qty

    if ask < prev_ask:
        ask_part = -ask_qty
    elif ask == prev_ask:
        ask_part = -(ask_qty - prev_ask_qty)
    else:
        ask_part = prev_ask_qty
    return bid_part + ask_part


@njit
def record_signal(
    hbt,
    flow,
    signal_ts,
    signal_bid,
    signal_ask,
    signal_bid_qty,
    signal_ask_qty,
    signal_buy_qty,
    signal_sell_qty,
    signal_buy_count,
    signal_sell_count,
    signal_ofi,
    write_idx,
    count,
):
    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        return write_idx, count
    ofi = 0.0
    if count > 0:
        prev_idx = (write_idx - 1) % SIGNAL_HISTORY_LEN
        ofi = pair_ofi(
            depth.best_bid,
            depth.best_ask,
            depth.best_bid_qty,
            depth.best_ask_qty,
            signal_bid[prev_idx],
            signal_ask[prev_idx],
            signal_bid_qty[prev_idx],
            signal_ask_qty[prev_idx],
        )
    signal_ts[write_idx] = hbt.current_timestamp
    signal_bid[write_idx] = depth.best_bid
    signal_ask[write_idx] = depth.best_ask
    signal_bid_qty[write_idx] = depth.best_bid_qty
    signal_ask_qty[write_idx] = depth.best_ask_qty
    signal_buy_qty[write_idx] = flow[2]
    signal_sell_qty[write_idx] = flow[3]
    signal_buy_count[write_idx] = flow[8]
    signal_sell_count[write_idx] = flow[9]
    signal_ofi[write_idx] = ofi
    write_idx = (write_idx + 1) % SIGNAL_HISTORY_LEN
    count = min(count + 1, SIGNAL_HISTORY_LEN)
    return write_idx, count


@njit
def signal_at_or_before(signal_ts, write_idx, count, target_ts):
    for offset in range(count):
        idx = (write_idx - 1 - offset) % SIGNAL_HISTORY_LEN
        if signal_ts[idx] <= target_ts:
            return idx
    return -1


@njit
def mid_from_history(signal_bid, signal_ask, idx):
    if idx < 0:
        return 0.0
    return (signal_bid[idx] + signal_ask[idx]) / 2.0


@njit
def recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count, window_ns):
    if count <= 1:
        return 0.0
    cur_idx = (write_idx - 1) % SIGNAL_HISTORY_LEN
    past_idx = signal_at_or_before(signal_ts, write_idx, count, signal_ts[cur_idx] - window_ns)
    if past_idx < 0:
        return 0.0
    return ratio_minus_one_bps(mid_from_history(signal_bid, signal_ask, cur_idx), mid_from_history(signal_bid, signal_ask, past_idx))


@njit
def recent_qty_change(signal_ts, signal_qty, write_idx, count, window_ns):
    if count <= 1:
        return 0.0
    cur_idx = (write_idx - 1) % SIGNAL_HISTORY_LEN
    past_idx = signal_at_or_before(signal_ts, write_idx, count, signal_ts[cur_idx] - window_ns)
    if past_idx < 0:
        return 0.0
    return signal_qty[cur_idx] - signal_qty[past_idx]


@njit
def rolling_signal_sum(signal_ts, signal_values, write_idx, count, window_ns):
    if count <= 0:
        return 0.0
    cur_idx = (write_idx - 1) % SIGNAL_HISTORY_LEN
    cutoff = signal_ts[cur_idx] - window_ns
    total = 0.0
    for offset in range(count):
        idx = (write_idx - 1 - offset) % SIGNAL_HISTORY_LEN
        if signal_ts[idx] < cutoff:
            break
        total += signal_values[idx]
    return total


@njit
def recent_vol_bps(signal_ts, signal_bid, signal_ask, write_idx, count, window_ns):
    if count <= 2:
        return 0.0
    cur_idx = (write_idx - 1) % SIGNAL_HISTORY_LEN
    cutoff = signal_ts[cur_idx] - window_ns
    total = 0.0
    n = 0
    prev_mid = 0.0
    have_prev = False
    for offset in range(count - 1, -1, -1):
        idx = (write_idx - 1 - offset) % SIGNAL_HISTORY_LEN
        if signal_ts[idx] < cutoff:
            continue
        mid = (signal_bid[idx] + signal_ask[idx]) / 2.0
        if have_prev and prev_mid > 0:
            total += abs(ratio_minus_one_bps(mid, prev_mid))
            n += 1
        prev_mid = mid
        have_prev = True
    if n <= 0:
        return 0.0
    return total / n


@njit
def microprice_bps(depth):
    mid = (depth.best_bid + depth.best_ask) / 2.0
    total_qty = depth.best_bid_qty + depth.best_ask_qty
    if total_qty <= 0 or mid <= 0:
        return 0.0
    micro = (depth.best_ask * depth.best_bid_qty + depth.best_bid * depth.best_ask_qty) / total_qty
    return ratio_minus_one_bps(micro, mid)


@njit
def queue_imbalance(depth):
    total_qty = depth.best_bid_qty + depth.best_ask_qty
    if total_qty <= 0:
        return 0.0
    return (depth.best_bid_qty - depth.best_ask_qty) / total_qty


@njit
def depth_qty(depth, side, level):
    if side > 0:
        return depth.bid_qty_at_tick(depth.best_bid_tick - level)
    return depth.ask_qty_at_tick(depth.best_ask_tick + level)


@njit
def depth_imbalance(depth, levels):
    bid_total = 0.0
    ask_total = 0.0
    for level in range(levels):
        bid_total += depth_qty(depth, 1, level)
        ask_total += depth_qty(depth, -1, level)
    total = bid_total + ask_total
    if total <= 0:
        return 0.0
    return (bid_total - ask_total) / total


@njit
def weighted_depth_imbalance_5(depth):
    bid_total = 0.0
    ask_total = 0.0
    for level in range(5):
        weight = 1.0 / (level + 1.0)
        bid_total += weight * depth_qty(depth, 1, level)
        ask_total += weight * depth_qty(depth, -1, level)
    total = bid_total + ask_total
    if total <= 0:
        return 0.0
    return (bid_total - ask_total) / total


@njit
def depth_slope_5(depth, side):
    top = depth_qty(depth, side, 0)
    far = depth_qty(depth, side, 4)
    total = 0.0
    for level in range(5):
        total += depth_qty(depth, side, level)
    if total <= 0:
        return 0.0
    return (top - far) / total


@njit
def update_trade_flow(hbt, flow):
    trades = hbt.last_trades(0)
    buy_qty = 0.0
    sell_qty = 0.0
    buy_count = 0.0
    sell_count = 0.0
    for trade in trades:
        if (trade.ev & BUY_EVENT) == BUY_EVENT:
            buy_qty += trade.qty
            buy_count += 1.0
        elif (trade.ev & SELL_EVENT) == SELL_EVENT:
            sell_qty += trade.qty
            sell_count += 1.0
    hbt.clear_last_trades(0)

    flow[0] = flow[0] * FLOW_DECAY + buy_qty
    flow[1] = flow[1] * FLOW_DECAY + sell_qty
    flow[2] = buy_qty
    flow[3] = sell_qty
    flow[4] += buy_qty
    flow[5] += sell_qty
    flow[6] += buy_count
    flow[7] += sell_count
    flow[8] = buy_count
    flow[9] = sell_count


@njit
def imbalance(buy_qty, sell_qty):
    total = buy_qty + sell_qty
    if total <= 0:
        return 0.0
    return (buy_qty - sell_qty) / total


@njit
def inventory_ratio(pos):
    if SOFT_POSITION_CONTRACTS <= 0:
        return 0.0
    return clamp(pos / SOFT_POSITION_CONTRACTS, -1.0, 1.0)


@njit
def factor_value(
    factor_id,
    depth,
    signal_ts,
    signal_bid,
    signal_ask,
    signal_bid_qty,
    signal_ask_qty,
    signal_buy_qty,
    signal_sell_qty,
    signal_buy_count,
    signal_sell_count,
    signal_ofi,
    write_idx,
    count,
    flow,
):
    if factor_id == 0:
        return ratio_minus_one_bps(depth.best_ask, depth.best_bid)
    if factor_id == 1:
        return queue_imbalance(depth)
    if factor_id == 2:
        return microprice_bps(depth)
    if factor_id == 3:
        return imbalance(flow[2], flow[3])
    if factor_id == 4:
        return imbalance(flow[0], flow[1])
    if factor_id == 5:
        return recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count, 100_000_000)
    if factor_id == 6:
        return recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count, 250_000_000)
    if factor_id == 7:
        return recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count, 1_000_000_000)
    if factor_id == 8:
        return abs(recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count, 1_000_000_000))
    if factor_id == 9:
        return depth_imbalance(depth, 3)
    if factor_id == 10:
        return depth_imbalance(depth, 5)
    if factor_id == 11:
        return weighted_depth_imbalance_5(depth)
    if factor_id == 12:
        return depth_slope_5(depth, 1)
    if factor_id == 13:
        return depth_slope_5(depth, -1)
    if factor_id == 14:
        return recent_qty_change(signal_ts, signal_bid_qty, write_idx, count, 100_000_000)
    if factor_id == 15:
        return recent_qty_change(signal_ts, signal_ask_qty, write_idx, count, 100_000_000)
    if factor_id == 16:
        if count <= 0:
            return 0.0
        return signal_ofi[(write_idx - 1) % SIGNAL_HISTORY_LEN]
    if factor_id == 17:
        return rolling_signal_sum(signal_ts, signal_ofi, write_idx, count, 1_000_000_000)
    if factor_id == 18:
        return rolling_signal_sum(signal_ts, signal_buy_qty, write_idx, count, 1_000_000_000) + rolling_signal_sum(
            signal_ts, signal_sell_qty, write_idx, count, 1_000_000_000
        )
    if factor_id == 19:
        return rolling_signal_sum(signal_ts, signal_buy_count, write_idx, count, 1_000_000_000) + rolling_signal_sum(
            signal_ts, signal_sell_count, write_idx, count, 1_000_000_000
        )
    if factor_id == 20:
        return recent_move_bps(signal_ts, signal_bid, signal_ask, write_idx, count, 3_000_000_000)
    if factor_id == 21:
        return recent_vol_bps(signal_ts, signal_bid, signal_ask, write_idx, count, 250_000_000)
    return 0.0


@njit
def predict_scores(
    depth,
    signal_ts,
    signal_bid,
    signal_ask,
    signal_bid_qty,
    signal_ask_qty,
    signal_buy_qty,
    signal_sell_qty,
    signal_buy_count,
    signal_sell_count,
    signal_ofi,
    write_idx,
    count,
    flow,
    model_mean,
    model_std,
    model_coef,
    include_interactions,
    clip_z,
):
    factor_count = len(model_mean)
    raw = np.empty(factor_count, dtype=np.float64)
    for i in range(factor_count):
        raw[i] = factor_value(
            i,
            depth,
            signal_ts,
            signal_bid,
            signal_ask,
            signal_bid_qty,
            signal_ask_qty,
            signal_buy_qty,
            signal_sell_qty,
            signal_buy_count,
            signal_sell_count,
            signal_ofi,
            write_idx,
            count,
            flow,
        )

    z = np.empty(factor_count, dtype=np.float64)
    for i in range(factor_count):
        value = (raw[i] - model_mean[i]) / model_std[i]
        if not math.isfinite(value):
            value = 0.0
        if value > clip_z:
            value = clip_z
        elif value < -clip_z:
            value = -clip_z
        z[i] = value

    features = np.empty(model_coef.shape[1] - 1, dtype=np.float64)
    idx = 0
    for i in range(factor_count):
        features[idx] = z[i]
        idx += 1
    for i in range(factor_count):
        features[idx] = z[i] * z[i]
        idx += 1
    if include_interactions:
        for i in range(factor_count):
            for j in range(i + 1, factor_count):
                features[idx] = z[i] * z[j]
                idx += 1
    while idx < len(features):
        features[idx] = 0.0
        idx += 1

    out = np.empty(6, dtype=np.float64)
    for model_idx in range(6):
        score = model_coef[model_idx, 0]
        for feature_idx in range(len(features)):
            score += model_coef[model_idx, feature_idx + 1] * features[feature_idx]
        if not math.isfinite(score):
            score = 0.0
        if model_idx == 4 or model_idx == 5:
            if score < 0.0:
                score = 0.0
            elif score > 1.0:
                score = 1.0
        out[model_idx] = score
    return out


@njit
def hist_bounds(score_idx):
    if score_idx == 0 or score_idx == 1:
        return EXPECTED_EDGE_HIST_MIN, EXPECTED_EDGE_HIST_MAX
    if score_idx == 2 or score_idx == 3:
        return EDGE_IF_FILLED_HIST_MIN, EDGE_IF_FILLED_HIST_MAX
    return FILL_PROB_HIST_MIN, FILL_PROB_HIST_MAX


@njit
def percentile_for_score(score_idx):
    if score_idx == 0 or score_idx == 1:
        return EXPECTED_EDGE_PERCENTILE
    if score_idx == 2 or score_idx == 3:
        return EDGE_IF_FILLED_PERCENTILE
    return FILL_PROB_PERCENTILE


@njit
def absolute_threshold_for_score(score_idx):
    if score_idx == 0 or score_idx == 1:
        return MIN_EXPECTED_EDGE_BPS
    if score_idx == 2 or score_idx == 3:
        return MIN_EDGE_IF_FILLED_BPS
    return MIN_FILL_PROB


@njit
def add_score_hist(hist_counts, score_seen, score_idx, value):
    if not math.isfinite(value):
        return
    low, high = hist_bounds(score_idx)
    if high <= low:
        return
    clipped = value
    if clipped < low:
        clipped = low
    elif clipped > high:
        clipped = high
    bin_idx = int((clipped - low) / (high - low) * PERCENTILE_BINS)
    if bin_idx < 0:
        bin_idx = 0
    elif bin_idx >= PERCENTILE_BINS:
        bin_idx = PERCENTILE_BINS - 1
    hist_counts[score_idx, bin_idx] += 1
    score_seen[score_idx] += 1


@njit
def hist_quantile(hist_counts, score_idx, percentile):
    low, high = hist_bounds(score_idx)
    total = 0
    for bin_idx in range(PERCENTILE_BINS):
        total += hist_counts[score_idx, bin_idx]
    if total <= 0:
        return absolute_threshold_for_score(score_idx)
    target = int(math.ceil(percentile * total))
    if target < 1:
        target = 1
    running = 0
    for bin_idx in range(PERCENTILE_BINS):
        running += hist_counts[score_idx, bin_idx]
        if running >= target:
            return low + (bin_idx + 0.5) / PERCENTILE_BINS * (high - low)
    return high


@njit
def update_percentile_gate(scores, hist_counts, score_seen, thresholds):
    for score_idx in range(6):
        add_score_hist(hist_counts, score_seen, score_idx, scores[score_idx])
        abs_threshold = absolute_threshold_for_score(score_idx)
        pct = percentile_for_score(score_idx)
        if USE_INTRADAY_PERCENTILE_GATE and pct > 0.0:
            if score_seen[score_idx] < PERCENTILE_WARMUP_SAMPLES:
                thresholds[score_idx] = 1e9
            else:
                pct_threshold = hist_quantile(hist_counts, score_idx, pct)
                thresholds[score_idx] = max(abs_threshold, pct_threshold)
        else:
            thresholds[score_idx] = abs_threshold


@njit
def is_reduce_side(side, pos):
    return (side > 0 and pos < 0) or (side < 0 and pos > 0)


@njit
def should_quote(side, pos, scores, thresholds, risk_halt_new_entries, regime_halt_new_entries):
    if ORDER_QTY < BITMEX_LOT_SIZE:
        return False, 1
    if side > 0 and pos + ORDER_QTY > MAX_POSITION_CONTRACTS:
        return False, 2
    if side < 0 and pos - ORDER_QTY < -MAX_POSITION_CONTRACTS:
        return False, 2
    reduce_side = is_reduce_side(side, pos)
    if risk_halt_new_entries and not reduce_side:
        return False, 4
    if regime_halt_new_entries and not reduce_side:
        return False, 6
    if REDUCE_ONLY_AFTER_SOFT_POSITION and abs(pos) >= SOFT_POSITION_CONTRACTS and not reduce_side:
        return False, 5
    if side > 0:
        expected = scores[0]
        edge_if_filled = scores[2]
        fill_prob = scores[4]
    else:
        expected = scores[1]
        edge_if_filled = scores[3]
        fill_prob = scores[5]
        expected_threshold = thresholds[1]
        edge_if_filled_threshold = thresholds[3]
        fill_prob_threshold = thresholds[5]
    if side > 0:
        expected_threshold = thresholds[0]
        edge_if_filled_threshold = thresholds[2]
        fill_prob_threshold = thresholds[4]
    matched = (
        expected >= expected_threshold + MIN_PLACEMENT_EXPECTED_MARGIN_BPS
        and edge_if_filled >= edge_if_filled_threshold + MIN_PLACEMENT_EDGE_IF_FILLED_MARGIN_BPS
        and fill_prob >= fill_prob_threshold
    )
    if not matched and not reduce_side:
        return False, 3
    return True, 0


@njit
def quote_half_spread_bps(side, depth, pos, scores, thresholds):
    mid = (depth.best_bid + depth.best_ask) / 2.0
    inv = inventory_ratio(pos)
    min_half_spread_bps = MIN_HALF_SPREAD_TICKS * BITMEX_TICK_SIZE / mid * 10_000.0
    vol_bps = 0.0
    half_spread = BASE_HALF_SPREAD_BPS + VOL_SPREAD_MULTIPLIER * vol_bps + abs(inv) * INVENTORY_SPREAD_BPS_AT_SOFT_LIMIT

    if USE_DYNAMIC_QUOTE and not is_reduce_side(side, pos):
        if side > 0:
            expected = scores[0]
            edge_if_filled = scores[2]
            fill_prob = scores[4]
            expected_threshold = thresholds[0]
            edge_if_filled_threshold = thresholds[2]
        else:
            expected = scores[1]
            edge_if_filled = scores[3]
            fill_prob = scores[5]
            expected_threshold = thresholds[1]
            edge_if_filled_threshold = thresholds[3]

        expected_margin = max(0.0, expected - expected_threshold)
        edge_if_filled_margin = max(0.0, edge_if_filled - edge_if_filled_threshold)
        tighten = (
            DYNAMIC_QUOTE_EXPECTED_EDGE_MULT * expected_margin
            + DYNAMIC_QUOTE_EDGE_IF_FILLED_MULT * edge_if_filled_margin
        )
        if tighten > DYNAMIC_QUOTE_MAX_TIGHTEN_BPS:
            tighten = DYNAMIC_QUOTE_MAX_TIGHTEN_BPS

        fill_prob_excess = max(0.0, fill_prob - DYNAMIC_QUOTE_FILL_PROB_BASELINE)
        widen = DYNAMIC_QUOTE_FILL_PROB_WIDEN_MULT * fill_prob_excess
        if widen > DYNAMIC_QUOTE_MAX_WIDEN_BPS:
            widen = DYNAMIC_QUOTE_MAX_WIDEN_BPS

        half_spread = half_spread - tighten + widen

    return max(half_spread, min_half_spread_bps)


@njit
def target_price(side, depth, pos, scores, thresholds):
    mid = (depth.best_bid + depth.best_ask) / 2.0
    inv = inventory_ratio(pos)
    half_spread = quote_half_spread_bps(side, depth, pos, scores, thresholds)
    anchor = mid * (1.0 - inv * INVENTORY_SKEW_BPS_AT_SOFT_LIMIT / 10_000.0)
    if side > 0:
        raw = anchor * (1.0 - half_spread / 10_000.0)
        return floor_to_tick(min(raw, depth.best_bid), BITMEX_TICK_SIZE)
    raw = anchor * (1.0 + half_spread / 10_000.0)
    return ceil_to_tick(max(raw, depth.best_ask), BITMEX_TICK_SIZE)


@njit
def update_placement_record(record, hbt, side, action, target_px, half_spread_bps, pos, scores, thresholds, metrics):
    record[0] = 1.0
    record[1] = action
    record[2] = hbt.current_timestamp
    record[3] = target_px
    record[4] = half_spread_bps
    record[5] = pos
    record[6] = scores[0]
    record[7] = scores[1]
    record[8] = scores[2]
    record[9] = scores[3]
    record[10] = scores[4]
    record[11] = scores[5]
    if side > 0:
        record[12] = scores[0]
        record[13] = scores[2]
        record[14] = scores[4]
        record[15] = thresholds[0]
        record[16] = thresholds[2]
        record[17] = thresholds[4]
    else:
        record[12] = scores[1]
        record[13] = scores[3]
        record[14] = scores[5]
        record[15] = thresholds[1]
        record[16] = thresholds[3]
        record[17] = thresholds[5]
    record[18] = record[12] - record[15]
    record[19] = record[13] - record[16]
    record[20] = record[14] - record[17]
    record[21] = metrics[57]
    record[22] = metrics[58]
    record[23] = side


@njit
def clear_placement_record(record):
    for idx in range(PLACEMENT_RECORD_LEN):
        record[idx] = 0.0


@njit
def manage_side(
    hbt,
    side,
    order_id,
    target_px,
    target_half_spread_bps,
    should_place,
    inflight_until,
    next_rest_allowed_ts,
    live_since,
    scores,
    thresholds,
    placement_record,
    metrics,
):
    if hbt.current_timestamp < inflight_until:
        return inflight_until, next_rest_allowed_ts, live_since

    existing = hbt.orders(0).get(order_id)
    rest_skip_metric = 18 if side > 0 else 19
    ttl_metric = 10 if side > 0 else 11
    filter_cancel_metric = 12 if side > 0 else 13
    modify_metric = 14 if side > 0 else 15
    place_metric = 16 if side > 0 else 17
    suppress_metric = 22 if side > 0 else 23

    if not should_place or target_px <= 0:
        metrics[suppress_metric] += 1
        if existing is not None and existing.cancellable:
            if hbt.current_timestamp < next_rest_allowed_ts:
                metrics[rest_skip_metric] += 1
                return inflight_until, next_rest_allowed_ts, live_since
            hbt.cancel(0, order_id, False)
            clear_placement_record(placement_record)
            metrics[filter_cancel_metric] += 1
            return hbt.current_timestamp + ORDER_INFLIGHT_NS, hbt.current_timestamp + REST_MIN_INTERVAL_NS, 0
        return inflight_until, next_rest_allowed_ts, live_since

    if existing is not None and live_since > 0 and hbt.current_timestamp - live_since >= ORDER_TTL_NS:
        if existing.cancellable:
            if hbt.current_timestamp < next_rest_allowed_ts:
                metrics[rest_skip_metric] += 1
                return inflight_until, next_rest_allowed_ts, live_since
            hbt.cancel(0, order_id, False)
            clear_placement_record(placement_record)
            metrics[ttl_metric] += 1
            return hbt.current_timestamp + ORDER_INFLIGHT_NS, hbt.current_timestamp + REST_MIN_INTERVAL_NS, 0
        return inflight_until, next_rest_allowed_ts, live_since

    if existing is not None:
        price_changed = abs(existing.price - target_px) >= MIN_AMEND_TICKS * BITMEX_TICK_SIZE
        if existing.cancellable and (price_changed or existing.qty != ORDER_QTY):
            if hbt.current_timestamp < next_rest_allowed_ts:
                metrics[rest_skip_metric] += 1
                return inflight_until, next_rest_allowed_ts, live_since
            hbt.modify(0, order_id, target_px, ORDER_QTY, False)
            update_placement_record(
                placement_record,
                hbt,
                side,
                2.0,
                target_px,
                target_half_spread_bps,
                hbt.position(0),
                scores,
                thresholds,
                metrics,
            )
            metrics[modify_metric] += 1
            return hbt.current_timestamp + ORDER_INFLIGHT_NS, hbt.current_timestamp + REST_MIN_INTERVAL_NS, hbt.current_timestamp
        return inflight_until, next_rest_allowed_ts, live_since

    if hbt.current_timestamp < next_rest_allowed_ts:
        metrics[rest_skip_metric] += 1
        return inflight_until, next_rest_allowed_ts, live_since

    if side > 0:
        hbt.submit_buy_order(0, order_id, target_px, ORDER_QTY, GTX, LIMIT, False)
    else:
        hbt.submit_sell_order(0, order_id, target_px, ORDER_QTY, GTX, LIMIT, False)
    update_placement_record(
        placement_record,
        hbt,
        side,
        1.0,
        target_px,
        target_half_spread_bps,
        hbt.position(0),
        scores,
        thresholds,
        metrics,
    )
    metrics[place_metric] += 1
    return hbt.current_timestamp + ORDER_INFLIGHT_NS, hbt.current_timestamp + REST_MIN_INTERVAL_NS, hbt.current_timestamp


@njit
def update_risk_metrics(hbt, metrics):
    depth = hbt.depth(0)
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        return
    mid = (depth.best_bid + depth.best_ask) / 2.0
    pos = abs(hbt.position(0))
    metrics[4] = max(metrics[4], pos)
    metrics[5] = max(metrics[5], pos * BITMEX_CONTRACT_SIZE)
    equity = bitmex_equity_usdt(hbt)
    if not math.isfinite(equity):
        return
    if metrics[47] == 0.0:
        metrics[47] = 1.0
        metrics[48] = equity
    if metrics[6] == 0.0 and metrics[7] == 0.0:
        metrics[6] = equity
        metrics[7] = equity
    metrics[6] = max(metrics[6], equity)
    metrics[7] = min(metrics[7], equity)


@njit
def risk_halt_new_entries(metrics):
    halted = False
    if DAILY_LOSS_LIMIT_USDT > 0.0 and metrics[47] > 0.0:
        if metrics[48] - metrics[7] >= DAILY_LOSS_LIMIT_USDT:
            halted = True
            metrics[49] += 1
    if DAILY_FILL_LIMIT > 0 and metrics[0] >= DAILY_FILL_LIMIT:
        halted = True
        metrics[50] += 1
    return halted


@njit
def update_regime_expected_edge_gate(scores, metrics):
    if not USE_REGIME_EXPECTED_EDGE_GATE:
        return False, False

    samples = metrics[59] + 1.0
    metrics[59] = samples
    alpha = REGIME_EXPECTED_EDGE_EWM_ALPHA
    if alpha <= 0.0:
        alpha = 1.0 / samples
    elif alpha > 1.0:
        alpha = 1.0

    if samples <= 1.0:
        metrics[57] = scores[0]
        metrics[58] = scores[1]
    else:
        metrics[57] = alpha * scores[0] + (1.0 - alpha) * metrics[57]
        metrics[58] = alpha * scores[1] + (1.0 - alpha) * metrics[58]

    metrics[62] += metrics[57]
    metrics[63] += metrics[58]

    if samples < REGIME_EXPECTED_EDGE_WARMUP_SAMPLES:
        return True, True

    bid_halt = metrics[57] < REGIME_MIN_BID_EXPECTED_EDGE_BPS
    ask_halt = metrics[58] < REGIME_MIN_ASK_EXPECTED_EDGE_BPS
    if bid_halt:
        metrics[55] += 1
    else:
        metrics[60] += 1
    if ask_halt:
        metrics[56] += 1
    else:
        metrics[61] += 1
    return bid_halt, ask_halt


@njit
def force_flatten(hbt, metrics):
    cancel_all_orders(hbt)
    hbt.elapse(1_000_000_000)
    pos = hbt.position(0)
    if abs(pos) <= 0:
        return
    before = bitmex_equity_usdt(hbt)
    depth = hbt.depth(0)
    if pos > 0:
        hbt.submit_sell_order(0, 91_001, depth.best_bid, abs(pos), IOC, MARKET, True)
    else:
        hbt.submit_buy_order(0, 91_002, depth.best_ask, abs(pos), IOC, MARKET, True)
    hbt.elapse(1_000_000_000)
    metrics[8] = bitmex_equity_usdt(hbt) - before


@njit
def run_strategy(
    hbt,
    recorder,
    metrics,
    fill_records,
    end_close_ts,
    model_mean,
    model_std,
    model_coef,
    include_interactions,
    clip_z,
):
    bid_id = 11_001
    ask_id = 21_001
    bid_inflight_until = 0
    ask_inflight_until = 0
    bid_live_since = 0
    ask_live_since = 0
    next_rest_allowed_ts = 0
    last_record_ts = 0
    last_pos = hbt.position(0)
    last_trades = hbt.state_values(0).num_trades
    last_trading_value = hbt.state_values(0).trading_value
    last_fill_equity = 0.0
    last_fill_equity_seen = False

    flow = np.zeros(10, dtype=np.float64)
    signal_ts = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.int64)
    signal_bid = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    signal_ask = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    signal_bid_qty = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    signal_ask_qty = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    signal_buy_qty = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    signal_sell_qty = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    signal_buy_count = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    signal_sell_count = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    signal_ofi = np.zeros(SIGNAL_HISTORY_LEN, dtype=np.float64)
    write_idx = 0
    count = 0
    hist_counts = np.zeros((6, PERCENTILE_BINS), dtype=np.int64)
    score_seen = np.zeros(6, dtype=np.float64)
    score_thresholds = np.array(
        [
            MIN_EXPECTED_EDGE_BPS,
            MIN_EXPECTED_EDGE_BPS,
            MIN_EDGE_IF_FILLED_BPS,
            MIN_EDGE_IF_FILLED_BPS,
            MIN_FILL_PROB,
            MIN_FILL_PROB,
        ],
        dtype=np.float64,
    )
    last_scores = np.zeros(6, dtype=np.float64)
    last_score_thresholds = np.array(
        [
            MIN_EXPECTED_EDGE_BPS,
            MIN_EXPECTED_EDGE_BPS,
            MIN_EDGE_IF_FILLED_BPS,
            MIN_EDGE_IF_FILLED_BPS,
            MIN_FILL_PROB,
            MIN_FILL_PROB,
        ],
        dtype=np.float64,
    )
    bid_placement_record = np.zeros(PLACEMENT_RECORD_LEN, dtype=np.float64)
    ask_placement_record = np.zeros(PLACEMENT_RECORD_LEN, dtype=np.float64)
    last_percentile_update_ts = 0

    while hbt.elapse(ORDER_UPDATE_INTERVAL_NS) == 0:
        if hbt.current_timestamp >= end_close_ts:
            break
        hbt.clear_inactive_orders(ALL_ASSETS)
        update_trade_flow(hbt, flow)
        metrics[24] = flow[4]
        metrics[25] = flow[5]
        metrics[26] = flow[6]
        metrics[27] = flow[7]
        write_idx, count = record_signal(
            hbt,
            flow,
            signal_ts,
            signal_bid,
            signal_ask,
            signal_bid_qty,
            signal_ask_qty,
            signal_buy_qty,
            signal_sell_qty,
            signal_buy_count,
            signal_sell_count,
            signal_ofi,
            write_idx,
            count,
        )

        depth = hbt.depth(0)
        if depth.best_bid <= 0 or depth.best_ask <= 0:
            cancel_all_orders(hbt)
            clear_placement_record(bid_placement_record)
            clear_placement_record(ask_placement_record)
        else:
            pos = hbt.position(0)
            scores = predict_scores(
                depth,
                signal_ts,
                signal_bid,
                signal_ask,
                signal_bid_qty,
                signal_ask_qty,
                signal_buy_qty,
                signal_sell_qty,
                signal_buy_count,
                signal_sell_count,
                signal_ofi,
                write_idx,
                count,
                flow,
                model_mean,
                model_std,
                model_coef,
                include_interactions,
                clip_z,
            )
            metrics[30] += scores[0]
            metrics[31] += scores[1]
            metrics[32] += scores[2]
            metrics[33] += scores[3]
            metrics[34] += scores[4]
            metrics[35] += scores[5]
            metrics[36] += 1
            for score_idx in range(6):
                last_scores[score_idx] = scores[score_idx]
            if USE_INTRADAY_PERCENTILE_GATE:
                if (
                    PERCENTILE_UPDATE_INTERVAL_NS <= 0
                    or hbt.current_timestamp - last_percentile_update_ts >= PERCENTILE_UPDATE_INTERVAL_NS
                ):
                    update_percentile_gate(scores, hist_counts, score_seen, score_thresholds)
                    last_percentile_update_ts = hbt.current_timestamp
                else:
                    for score_idx in range(6):
                        add_score_hist(hist_counts, score_seen, score_idx, scores[score_idx])
            if (
                score_thresholds[0] < 1e8
                and score_thresholds[1] < 1e8
                and score_thresholds[2] < 1e8
                and score_thresholds[3] < 1e8
            ):
                metrics[40] += score_thresholds[0]
                metrics[41] += score_thresholds[1]
                metrics[42] += score_thresholds[2]
                metrics[43] += score_thresholds[3]
                metrics[44] += score_thresholds[4]
                metrics[45] += score_thresholds[5]
                metrics[46] += 1
            for score_idx in range(6):
                last_score_thresholds[score_idx] = score_thresholds[score_idx]
            update_risk_metrics(hbt, metrics)
            risk_halt = risk_halt_new_entries(metrics)
            bid_regime_halt, ask_regime_halt = update_regime_expected_edge_gate(scores, metrics)
            bid_ok, bid_reason = should_quote(1, pos, scores, score_thresholds, risk_halt, bid_regime_halt)
            ask_ok, ask_reason = should_quote(-1, pos, scores, score_thresholds, risk_halt, ask_regime_halt)
            if bid_reason == 3:
                metrics[20] += 1
            elif bid_reason == 4:
                metrics[51] += 1
            elif bid_reason == 5:
                metrics[53] += 1
            elif bid_ok:
                metrics[38] += 1
            if ask_reason == 3:
                metrics[21] += 1
            elif ask_reason == 4:
                metrics[52] += 1
            elif ask_reason == 5:
                metrics[54] += 1
            elif ask_ok:
                metrics[39] += 1

            bid_half_spread = quote_half_spread_bps(1, depth, pos, scores, score_thresholds)
            ask_half_spread = quote_half_spread_bps(-1, depth, pos, scores, score_thresholds)
            metrics[65] += bid_half_spread
            metrics[66] += ask_half_spread
            metrics[67] += 1
            metrics[68] += 1

            bid_px = target_price(1, depth, pos, scores, score_thresholds)
            bid_inflight_until, next_rest_allowed_ts, bid_live_since = manage_side(
                hbt,
                1,
                bid_id,
                bid_px,
                bid_half_spread,
                bid_ok,
                bid_inflight_until,
                next_rest_allowed_ts,
                bid_live_since,
                scores,
                score_thresholds,
                bid_placement_record,
                metrics,
            )
            ask_px = target_price(-1, depth, pos, scores, score_thresholds)
            ask_inflight_until, next_rest_allowed_ts, ask_live_since = manage_side(
                hbt,
                -1,
                ask_id,
                ask_px,
                ask_half_spread,
                ask_ok,
                ask_inflight_until,
                next_rest_allowed_ts,
                ask_live_since,
                scores,
                score_thresholds,
                ask_placement_record,
                metrics,
            )

        state = hbt.state_values(0)
        if state.num_trades > last_trades:
            depth = hbt.depth(0)
            if depth.best_bid > 0 and depth.best_ask > 0:
                mid = (depth.best_bid + depth.best_ask) / 2.0
                delta_contracts = state.position - last_pos
                delta_value = state.trading_value - last_trading_value
                exec_px = 0.0
                if abs(delta_contracts) > 0 and delta_value > 0:
                    exec_px = delta_value / (abs(delta_contracts) * BITMEX_CONTRACT_SIZE)
                fill_count = state.num_trades - last_trades
                metrics[0] += fill_count
                metrics[3] += abs(delta_contracts) * BITMEX_CONTRACT_SIZE
                if delta_contracts > 0:
                    metrics[1] += fill_count
                    if exec_px > 0:
                        metrics[28] += mid - exec_px
                        metrics[29] += 1
                elif delta_contracts < 0:
                    metrics[2] += fill_count
                    if exec_px > 0:
                        metrics[28] += exec_px - mid
                        metrics[29] += 1
                fill_record_idx = int(metrics[64])
                if fill_record_idx < fill_records.shape[0]:
                    side = 0.0
                    spread_capture = 0.0
                    side_expected = 0.0
                    side_edge_if_filled = 0.0
                    side_fill_prob = 0.0
                    side_expected_threshold = 0.0
                    side_edge_if_filled_threshold = 0.0
                    side_fill_prob_threshold = 0.0
                    if delta_contracts > 0:
                        side = 1.0
                        if exec_px > 0:
                            spread_capture = mid - exec_px
                        side_expected = last_scores[0]
                        side_edge_if_filled = last_scores[2]
                        side_fill_prob = last_scores[4]
                        side_expected_threshold = last_score_thresholds[0]
                        side_edge_if_filled_threshold = last_score_thresholds[2]
                        side_fill_prob_threshold = last_score_thresholds[4]
                    elif delta_contracts < 0:
                        side = -1.0
                        if exec_px > 0:
                            spread_capture = exec_px - mid
                        side_expected = last_scores[1]
                        side_edge_if_filled = last_scores[3]
                        side_fill_prob = last_scores[5]
                        side_expected_threshold = last_score_thresholds[1]
                        side_edge_if_filled_threshold = last_score_thresholds[3]
                        side_fill_prob_threshold = last_score_thresholds[5]
                    equity = bitmex_equity_usdt(hbt)
                    if last_fill_equity_seen:
                        equity_delta = equity - last_fill_equity
                    elif metrics[47] > 0.0:
                        equity_delta = equity - metrics[48]
                    else:
                        equity_delta = 0.0
                    last_fill_equity = equity
                    last_fill_equity_seen = True

                    fill_records[fill_record_idx, 0] = hbt.current_timestamp
                    fill_records[fill_record_idx, 1] = side
                    fill_records[fill_record_idx, 2] = fill_count
                    fill_records[fill_record_idx, 3] = delta_contracts
                    fill_records[fill_record_idx, 4] = last_pos
                    fill_records[fill_record_idx, 5] = state.position
                    fill_records[fill_record_idx, 6] = exec_px
                    fill_records[fill_record_idx, 7] = mid
                    fill_records[fill_record_idx, 8] = spread_capture
                    fill_records[fill_record_idx, 9] = equity
                    fill_records[fill_record_idx, 10] = equity_delta
                    fill_records[fill_record_idx, 11] = last_scores[0]
                    fill_records[fill_record_idx, 12] = last_scores[1]
                    fill_records[fill_record_idx, 13] = last_scores[2]
                    fill_records[fill_record_idx, 14] = last_scores[3]
                    fill_records[fill_record_idx, 15] = last_scores[4]
                    fill_records[fill_record_idx, 16] = last_scores[5]
                    fill_records[fill_record_idx, 17] = side_expected
                    fill_records[fill_record_idx, 18] = side_edge_if_filled
                    fill_records[fill_record_idx, 19] = side_fill_prob
                    fill_records[fill_record_idx, 20] = side_expected_threshold
                    fill_records[fill_record_idx, 21] = side_edge_if_filled_threshold
                    fill_records[fill_record_idx, 22] = side_fill_prob_threshold
                    fill_records[fill_record_idx, 23] = side_expected - side_expected_threshold
                    fill_records[fill_record_idx, 24] = side_edge_if_filled - side_edge_if_filled_threshold
                    fill_records[fill_record_idx, 25] = side_fill_prob - side_fill_prob_threshold
                    fill_records[fill_record_idx, 26] = metrics[57]
                    fill_records[fill_record_idx, 27] = metrics[58]
                    if delta_contracts > 0:
                        placement = bid_placement_record
                    else:
                        placement = ask_placement_record
                    placement_age = 0.0
                    if placement[0] > 0.0:
                        placement_age = hbt.current_timestamp - placement[2]
                    fill_records[fill_record_idx, 28] = placement[0]
                    fill_records[fill_record_idx, 29] = placement[1]
                    fill_records[fill_record_idx, 30] = placement[2]
                    fill_records[fill_record_idx, 31] = placement_age
                    fill_records[fill_record_idx, 32] = placement[3]
                    fill_records[fill_record_idx, 33] = placement[4]
                    fill_records[fill_record_idx, 34] = placement[5]
                    fill_records[fill_record_idx, 35] = placement[6]
                    fill_records[fill_record_idx, 36] = placement[7]
                    fill_records[fill_record_idx, 37] = placement[8]
                    fill_records[fill_record_idx, 38] = placement[9]
                    fill_records[fill_record_idx, 39] = placement[10]
                    fill_records[fill_record_idx, 40] = placement[11]
                    fill_records[fill_record_idx, 41] = placement[12]
                    fill_records[fill_record_idx, 42] = placement[13]
                    fill_records[fill_record_idx, 43] = placement[14]
                    fill_records[fill_record_idx, 44] = placement[15]
                    fill_records[fill_record_idx, 45] = placement[16]
                    fill_records[fill_record_idx, 46] = placement[17]
                    fill_records[fill_record_idx, 47] = placement[18]
                    fill_records[fill_record_idx, 48] = placement[19]
                    fill_records[fill_record_idx, 49] = placement[20]
                    fill_records[fill_record_idx, 50] = placement[21]
                    fill_records[fill_record_idx, 51] = placement[22]
                    metrics[64] += 1
            last_pos = state.position
            last_trades = state.num_trades
            last_trading_value = state.trading_value

        update_risk_metrics(hbt, metrics)
        if hbt.current_timestamp - last_record_ts >= 1_000_000_000:
            recorder.record(hbt)
            last_record_ts = hbt.current_timestamp

    force_flatten(hbt, metrics)
    recorder.record(hbt)
    return True


def build_asset(bitmex_npz: Path):
    asset = (
        BacktestAsset()
        .data([str(bitmex_npz)])
        .linear_asset(BITMEX_CONTRACT_SIZE)
        .constant_order_latency(BITMEX_ORDER_ENTRY_LATENCY_NS, BITMEX_ORDER_RESPONSE_LATENCY_NS)
        .risk_adverse_queue_model()
        .trading_value_fee_model(MAKER_FEE_RATE, TAKER_FEE_RATE)
        .tick_size(BITMEX_TICK_SIZE)
        .lot_size(BITMEX_LOT_SIZE)
        .last_trades_capacity(10_000)
    )
    if EXCHANGE_MODEL == "partial":
        return asset.partial_fill_exchange()
    if EXCHANGE_MODEL == "strict_no_partial":
        return asset.strict_no_partial_fill_exchange()
    return asset.no_partial_fill_exchange()


def find_model_file(model_dir: Path, model_tag: str, yyyymmdd: str, explicit_model_file: str = "") -> Path:
    if explicit_model_file:
        path = Path(explicit_model_file)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    pattern = f"bitmex_xbtusdt_{model_tag}_train_*_test_{yyyymmdd}.edge_model.json"
    matches = sorted(model_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no model found: {model_dir / pattern}")
    if len(matches) > 1:
        print(f"WARNING: multiple models for {yyyymmdd}; using {matches[-1]}")
    return matches[-1]


def load_edge_model(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, float, dict]:
    model = json.loads(path.read_text())
    mean = np.array(model["mean"], dtype=np.float64)
    std = np.array(model["std"], dtype=np.float64)
    include_interactions = bool(model.get("include_interactions", True))
    clip_z = float(model.get("clip_z", 6.0))
    factor_count = len(mean)
    feature_len = factor_count * 2 + (factor_count * (factor_count - 1) // 2 if include_interactions else 0)
    coef = np.zeros((6, feature_len + 1), dtype=np.float64)
    for idx, key in enumerate(MODEL_KEYS):
        item = model.get("models", {}).get(key)
        if item is None:
            continue
        raw_coef = np.array(item["coef"], dtype=np.float64)
        limit = min(len(raw_coef), feature_len + 1)
        coef[idx, :limit] = raw_coef[:limit]
    return mean, std, coef, include_interactions, clip_z, model


def load_rules(
    path: Path,
    min_expected_edge_bps: float,
    min_edge_if_filled_bps: float,
    min_fill_prob: float,
    min_fill_samples: int,
    max_rules_per_side: int,
    min_factor_matches: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rows = []
    with path.open() as file:
        for row in csv.DictReader(file):
            factors: list[str]
            mins: list[float]
            maxs: list[float]
            if "factor1" in row and row.get("factor1") and row.get("factor2"):
                factors = [row["factor1"], row["factor2"]]
                mins = [float(row["factor1_min"]), float(row["factor2_min"])]
                maxs = [float(row["factor1_max"]), float(row["factor2_max"])]
            else:
                factor = row["factor"]
                factors = [factor]
                mins = [float(row["factor_min"])]
                maxs = [float(row["factor_max"])]
            if len(factors) < min_factor_matches:
                continue
            if any(factor not in FACTOR_IDS for factor in factors):
                continue
            expected = float(row["expected_edge_bps"])
            edge = float(row["edge_if_filled_bps"])
            fill_prob = float(row["fill_prob"])
            fill_samples = int(float(row["fill_samples"]))
            if expected < min_expected_edge_bps:
                continue
            if edge < min_edge_if_filled_bps:
                continue
            if fill_prob < min_fill_prob:
                continue
            if fill_samples < min_fill_samples:
                continue
            rows.append(
                {
                    "factor": "&".join(factors),
                    "factors": factors,
                    "factor_ids": [FACTOR_IDS[factor] for factor in factors],
                    "factor_mins": mins,
                    "factor_maxs": maxs,
                    "factor_count": len(factors),
                    "side": row["side"],
                    "side_id": 1 if row["side"] == "bid" else -1,
                    "factor_min": mins[0],
                    "factor_max": maxs[0],
                    "factor1": factors[0],
                    "factor1_min": mins[0],
                    "factor1_max": maxs[0],
                    "factor2": factors[1] if len(factors) > 1 else "",
                    "factor2_min": mins[1] if len(factors) > 1 else float("nan"),
                    "factor2_max": maxs[1] if len(factors) > 1 else float("nan"),
                    "expected_edge_bps": expected,
                    "edge_if_filled_bps": edge,
                    "fill_prob": fill_prob,
                    "fill_samples": fill_samples,
                }
            )
    selected = []
    for side in ("bid", "ask"):
        side_rows = [row for row in rows if row["side"] == side]
        side_rows.sort(key=lambda row: row["expected_edge_bps"], reverse=True)
        selected.extend(side_rows[:max_rules_per_side])
    selected.sort(key=lambda row: row["expected_edge_bps"], reverse=True)
    if not selected:
        raise ValueError(f"no rules selected from {path}")
    max_legs = max(len(row["factor_ids"]) for row in selected)
    rule_factor_ids = np.full((len(selected), max_legs), -1, dtype=np.int64)
    rule_mins = np.full((len(selected), max_legs), np.nan, dtype=np.float64)
    rule_maxs = np.full((len(selected), max_legs), np.nan, dtype=np.float64)
    rule_leg_counts = np.zeros(len(selected), dtype=np.int64)
    for row_idx, row in enumerate(selected):
        factor_ids = row["factor_ids"]
        mins = row["factor_mins"]
        maxs = row["factor_maxs"]
        rule_leg_counts[row_idx] = len(factor_ids)
        for leg_idx, factor_id in enumerate(factor_ids):
            rule_factor_ids[row_idx, leg_idx] = factor_id
            rule_mins[row_idx, leg_idx] = mins[leg_idx]
            rule_maxs[row_idx, leg_idx] = maxs[leg_idx]
    return (
        rule_factor_ids,
        np.array([row["side_id"] for row in selected], dtype=np.int64),
        rule_mins,
        rule_maxs,
        rule_leg_counts,
        selected,
    )


def rule_factor_coverage(selected_rules: list[dict]) -> dict:
    bid_factors = sorted({row["factor"] for row in selected_rules if row["side"] == "bid"})
    ask_factors = sorted({row["factor"] for row in selected_rules if row["side"] == "ask"})
    bid_atomic_factors = sorted({factor for row in selected_rules if row["side"] == "bid" for factor in row["factors"]})
    ask_atomic_factors = sorted({factor for row in selected_rules if row["side"] == "ask" for factor in row["factors"]})
    bid_rule_factors = [len(row["factors"]) for row in selected_rules if row["side"] == "bid"]
    ask_rule_factors = [len(row["factors"]) for row in selected_rules if row["side"] == "ask"]
    return {
        "bid": bid_factors,
        "ask": ask_factors,
        "bid_count": len(bid_factors),
        "ask_count": len(ask_factors),
        "bid_atomic_factors": bid_atomic_factors,
        "ask_atomic_factors": ask_atomic_factors,
        "bid_atomic_count": len(bid_atomic_factors),
        "ask_atomic_count": len(ask_atomic_factors),
        "bid_max_rule_factors": max(bid_rule_factors) if bid_rule_factors else 0,
        "ask_max_rule_factors": max(ask_rule_factors) if ask_rule_factors else 0,
    }


def run_backtest(bitmex_npz: Path, yyyymmdd: str, model_path: Path) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    model_mean, model_std, model_coef, include_interactions, clip_z, model = load_edge_model(model_path)
    hbt = HashMapMarketDepthBacktest([build_asset(bitmex_npz)])
    recorder = Recorder(1, 100_000)
    metrics = np.zeros(72, dtype=np.float64)
    fill_records = np.zeros((20_000, len(FILL_ATTRIBUTION_FIELDS) - 2), dtype=np.float64)
    ok = run_strategy(
        hbt,
        recorder.recorder,
        metrics,
        fill_records,
        end_close_ts_ns(yyyymmdd),
        model_mean,
        model_std,
        model_coef,
        include_interactions,
        clip_z,
    )
    if not ok:
        raise RuntimeError("strategy returned false")

    tag = RESULT_TAG or f"edge_scored_{EXCHANGE_MODEL}"
    out = RESULT_DIR / f"bitmex_xbtusdt_edge_scored_maker_{tag}_{yyyymmdd}.npz"
    fill_count = int(metrics[64])
    np.savez_compressed(out, **{"0": recorder.get(0), "metrics": metrics, "fill_records": fill_records[:fill_count]})
    write_summary(out, yyyymmdd, model_path, model)
    write_fill_attribution(out, yyyymmdd)
    return out


def signed_money(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.4f} USDT"


def write_summary(result_npz: Path, yyyymmdd: str, model_path: Path, model: dict) -> dict:
    data = np.load(result_npz)
    records = data["0"]
    metrics = data["metrics"]
    final = records[-1]
    final_price = float(final["price"])
    final_pos = float(final["position"])
    equity = float(final["balance"]) + final_pos * final_price * BITMEX_CONTRACT_SIZE - float(final["fee"])
    total_fee = float(final["fee"])
    gross = equity + total_fee
    avg_capture = metrics[28] / metrics[29] if metrics[29] > 0 else 0.0
    score_samples = metrics[36] if metrics[36] > 0 else 1.0
    threshold_samples = metrics[46] if metrics[46] > 0 else 1.0
    regime_samples = metrics[59] if metrics[59] > 0 else 1.0
    bid_quote_samples = metrics[67] if metrics[67] > 0 else 1.0
    ask_quote_samples = metrics[68] if metrics[68] > 0 else 1.0
    summary = {
        "date": yyyymmdd,
        "symbol": BITMEX_SYMBOL,
        "strategy": "edge_scored_maker",
        "exchange_model": EXCHANGE_MODEL,
        "maker_fee_rate": MAKER_FEE_RATE,
        "taker_fee_rate": TAKER_FEE_RATE,
        "order_qty_contracts": ORDER_QTY,
        "base_half_spread_bps": BASE_HALF_SPREAD_BPS,
        "max_position_contracts": MAX_POSITION_CONTRACTS,
        "soft_position_contracts": SOFT_POSITION_CONTRACTS,
        "dynamic_quote": USE_DYNAMIC_QUOTE,
        "dynamic_quote_expected_edge_mult": DYNAMIC_QUOTE_EXPECTED_EDGE_MULT,
        "dynamic_quote_edge_if_filled_mult": DYNAMIC_QUOTE_EDGE_IF_FILLED_MULT,
        "dynamic_quote_max_tighten_bps": DYNAMIC_QUOTE_MAX_TIGHTEN_BPS,
        "dynamic_quote_fill_prob_widen_mult": DYNAMIC_QUOTE_FILL_PROB_WIDEN_MULT,
        "dynamic_quote_fill_prob_baseline": DYNAMIC_QUOTE_FILL_PROB_BASELINE,
        "dynamic_quote_max_widen_bps": DYNAMIC_QUOTE_MAX_WIDEN_BPS,
        "order_ttl_ms": ORDER_TTL_NS / 1_000_000.0,
        "rest_min_interval_ms": REST_MIN_INTERVAL_NS / 1_000_000.0,
        "model_path": str(model_path),
        "model_horizon_ms": model.get("horizon_ms", ""),
        "min_expected_edge_bps": MIN_EXPECTED_EDGE_BPS,
        "min_edge_if_filled_bps": MIN_EDGE_IF_FILLED_BPS,
        "min_fill_prob": MIN_FILL_PROB,
        "min_placement_expected_margin_bps": MIN_PLACEMENT_EXPECTED_MARGIN_BPS,
        "min_placement_edge_if_filled_margin_bps": MIN_PLACEMENT_EDGE_IF_FILLED_MARGIN_BPS,
        "reduce_only_after_soft_position": REDUCE_ONLY_AFTER_SOFT_POSITION,
        "daily_loss_limit_usdt": DAILY_LOSS_LIMIT_USDT,
        "daily_fill_limit": DAILY_FILL_LIMIT,
        "regime_expected_edge_gate": USE_REGIME_EXPECTED_EDGE_GATE,
        "regime_expected_edge_warmup_samples": REGIME_EXPECTED_EDGE_WARMUP_SAMPLES,
        "regime_expected_edge_ewm_alpha": REGIME_EXPECTED_EDGE_EWM_ALPHA,
        "regime_min_bid_expected_edge_bps": REGIME_MIN_BID_EXPECTED_EDGE_BPS,
        "regime_min_ask_expected_edge_bps": REGIME_MIN_ASK_EXPECTED_EDGE_BPS,
        "intraday_percentile_gate": USE_INTRADAY_PERCENTILE_GATE,
        "expected_edge_percentile": EXPECTED_EDGE_PERCENTILE,
        "edge_if_filled_percentile": EDGE_IF_FILLED_PERCENTILE,
        "fill_prob_percentile": FILL_PROB_PERCENTILE,
        "percentile_warmup_samples": PERCENTILE_WARMUP_SAMPLES,
        "percentile_update_interval_ms": PERCENTILE_UPDATE_INTERVAL_NS / 1_000_000.0,
        "total_pnl_usdt": equity,
        "gross_pnl_before_fee_usdt": gross,
        "fee_usdt": total_fee,
        "maker_rebate_usdt": -total_fee,
        "fills": int(metrics[0]),
        "buy_fills": int(metrics[1]),
        "sell_fills": int(metrics[2]),
        "filled_base_btc": float(metrics[3]),
        "max_position_contracts_seen": float(metrics[4]),
        "max_position_btc_seen": float(metrics[5]),
        "max_equity_usdt": float(metrics[6]),
        "min_equity_usdt": float(metrics[7]),
        "force_close_pnl_usdt": float(metrics[8]),
        "initial_equity_usdt": float(metrics[48]),
        "risk_halt_gate_bid": int(metrics[51]),
        "risk_halt_gate_ask": int(metrics[52]),
        "soft_position_gate_bid": int(metrics[53]),
        "soft_position_gate_ask": int(metrics[54]),
        "regime_gate_bid": int(metrics[55]),
        "regime_gate_ask": int(metrics[56]),
        "final_regime_bid_expected_edge_ewm_bps": float(metrics[57]),
        "final_regime_ask_expected_edge_ewm_bps": float(metrics[58]),
        "regime_samples": int(metrics[59]),
        "regime_allow_bid": int(metrics[60]),
        "regime_allow_ask": int(metrics[61]),
        "avg_regime_bid_expected_edge_ewm_bps": float(metrics[62] / regime_samples),
        "avg_regime_ask_expected_edge_ewm_bps": float(metrics[63] / regime_samples),
        "ttl_cancel_bid": int(metrics[10]),
        "ttl_cancel_ask": int(metrics[11]),
        "filter_cancel_bid": int(metrics[12]),
        "filter_cancel_ask": int(metrics[13]),
        "modify_bid": int(metrics[14]),
        "modify_ask": int(metrics[15]),
        "place_bid": int(metrics[16]),
        "place_ask": int(metrics[17]),
        "rest_skip_bid": int(metrics[18]),
        "rest_skip_ask": int(metrics[19]),
        "score_gate_bid": int(metrics[20]),
        "score_gate_ask": int(metrics[21]),
        "suppress_bid": int(metrics[22]),
        "suppress_ask": int(metrics[23]),
        "market_buy_qty_seen": float(metrics[24]),
        "market_sell_qty_seen": float(metrics[25]),
        "market_buy_trade_count_seen": int(metrics[26]),
        "market_sell_trade_count_seen": int(metrics[27]),
        "avg_spread_capture_usdt_per_btc": float(avg_capture),
        "spread_capture_events": int(metrics[29]),
        "avg_pred_bid_expected_edge_bps": float(metrics[30] / score_samples),
        "avg_pred_ask_expected_edge_bps": float(metrics[31] / score_samples),
        "avg_pred_bid_edge_if_filled_bps": float(metrics[32] / score_samples),
        "avg_pred_ask_edge_if_filled_bps": float(metrics[33] / score_samples),
        "avg_pred_bid_fill_prob": float(metrics[34] / score_samples),
        "avg_pred_ask_fill_prob": float(metrics[35] / score_samples),
        "score_samples": int(metrics[36]),
        "score_pass_bid": int(metrics[38]),
        "score_pass_ask": int(metrics[39]),
        "avg_gate_bid_expected_edge_bps": float(metrics[40] / threshold_samples),
        "avg_gate_ask_expected_edge_bps": float(metrics[41] / threshold_samples),
        "avg_gate_bid_edge_if_filled_bps": float(metrics[42] / threshold_samples),
        "avg_gate_ask_edge_if_filled_bps": float(metrics[43] / threshold_samples),
        "avg_gate_bid_fill_prob": float(metrics[44] / threshold_samples),
        "avg_gate_ask_fill_prob": float(metrics[45] / threshold_samples),
        "avg_quote_bid_half_spread_bps": float(metrics[65] / bid_quote_samples),
        "avg_quote_ask_half_spread_bps": float(metrics[66] / ask_quote_samples),
        "final_position_contracts": final_pos,
        "final_position_btc": final_pos * BITMEX_CONTRACT_SIZE,
        "records": int(len(records)),
        "num_trades": int(final["num_trades"]),
    }
    result_npz.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(render_console_summary(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def render_console_summary(summary: dict) -> str:
    return "\n".join(
        [
            "",
            "========== BitMEX edge-scored maker 回测 ==========",
            f"日期: {summary['date']}",
            f"PnL: {signed_money(summary['total_pnl_usdt'])}",
            f"gross 不含手续费: {signed_money(summary['gross_pnl_before_fee_usdt'])}",
            f"手续费/返佣贡献: {signed_money(summary['maker_rebate_usdt'])}",
            f"fills: {summary['fills']}，买={summary['buy_fills']}，卖={summary['sell_fills']}",
            f"最大仓位: {summary['max_position_contracts_seen']:,.0f} contracts",
            f"score gate: bid={summary['score_gate_bid']}, ask={summary['score_gate_ask']}",
            f"risk gate: bid={summary['risk_halt_gate_bid']}, ask={summary['risk_halt_gate_ask']}",
            f"regime gate: bid={summary['regime_gate_bid']}, ask={summary['regime_gate_ask']}",
            f"soft position gate: bid={summary['soft_position_gate_bid']}, ask={summary['soft_position_gate_ask']}",
            f"score pass: bid={summary['score_pass_bid']}, ask={summary['score_pass_ask']}",
            (
                "avg gate thresholds: "
                f"bid_exp={summary['avg_gate_bid_expected_edge_bps']:.4f}, "
                f"ask_exp={summary['avg_gate_ask_expected_edge_bps']:.4f}, "
                f"bid_if={summary['avg_gate_bid_edge_if_filled_bps']:.4f}, "
                f"ask_if={summary['avg_gate_ask_edge_if_filled_bps']:.4f}, "
                f"bid_fill={summary['avg_gate_bid_fill_prob']:.4f}, "
                f"ask_fill={summary['avg_gate_ask_fill_prob']:.4f}"
            ),
            (
                "avg quote half-spread: "
                f"bid={summary['avg_quote_bid_half_spread_bps']:.4f}bps, "
                f"ask={summary['avg_quote_ask_half_spread_bps']:.4f}bps"
            ),
            f"filter cancel: bid={summary['filter_cancel_bid']}, ask={summary['filter_cancel_ask']}",
            f"avg capture: {summary['avg_spread_capture_usdt_per_btc']:,.4f} USDT/BTC",
            "=======================================================",
            "",
        ]
    )


def write_fill_attribution(result_npz: Path, yyyymmdd: str) -> Path:
    data = np.load(result_npz)
    rows = data["fill_records"] if "fill_records" in data else np.zeros((0, len(FILL_ATTRIBUTION_FIELDS) - 2))
    out = result_npz.with_suffix(".fills.csv")
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FILL_ATTRIBUTION_FIELDS)
        writer.writeheader()
        for idx, values in enumerate(rows):
            writer.writerow(
                {
                    "date": yyyymmdd,
                    "fill_index": idx,
                    "timestamp_ns": int(values[0]),
                    "side": "bid" if values[1] > 0 else "ask" if values[1] < 0 else "",
                    "fill_count": int(values[2]),
                    "delta_contracts": values[3],
                    "position_before": values[4],
                    "position_after": values[5],
                    "exec_px": values[6],
                    "mid": values[7],
                    "spread_capture_usdt_per_btc": values[8],
                    "equity_usdt": values[9],
                    "equity_delta_since_prev_fill_usdt": values[10],
                    "bid_expected_edge_bps": values[11],
                    "ask_expected_edge_bps": values[12],
                    "bid_edge_if_filled_bps": values[13],
                    "ask_edge_if_filled_bps": values[14],
                    "bid_fill_prob": values[15],
                    "ask_fill_prob": values[16],
                    "side_expected_edge_bps": values[17],
                    "side_edge_if_filled_bps": values[18],
                    "side_fill_prob": values[19],
                    "side_expected_threshold_bps": values[20],
                    "side_edge_if_filled_threshold_bps": values[21],
                    "side_fill_prob_threshold": values[22],
                    "side_expected_margin_bps": values[23],
                    "side_edge_if_filled_margin_bps": values[24],
                    "side_fill_prob_margin": values[25],
                    "bid_regime_expected_edge_ewm_bps": values[26],
                    "ask_regime_expected_edge_ewm_bps": values[27],
                    "placement_valid": int(values[28]),
                    "placement_action": "submit" if values[29] == 1.0 else "modify" if values[29] == 2.0 else "",
                    "placement_timestamp_ns": int(values[30]),
                    "placement_age_ns": int(values[31]),
                    "placement_target_px": values[32],
                    "placement_half_spread_bps": values[33],
                    "placement_position_contracts": values[34],
                    "placement_bid_expected_edge_bps": values[35],
                    "placement_ask_expected_edge_bps": values[36],
                    "placement_bid_edge_if_filled_bps": values[37],
                    "placement_ask_edge_if_filled_bps": values[38],
                    "placement_bid_fill_prob": values[39],
                    "placement_ask_fill_prob": values[40],
                    "placement_side_expected_edge_bps": values[41],
                    "placement_side_edge_if_filled_bps": values[42],
                    "placement_side_fill_prob": values[43],
                    "placement_side_expected_threshold_bps": values[44],
                    "placement_side_edge_if_filled_threshold_bps": values[45],
                    "placement_side_fill_prob_threshold": values[46],
                    "placement_side_expected_margin_bps": values[47],
                    "placement_side_edge_if_filled_margin_bps": values[48],
                    "placement_side_fill_prob_margin": values[49],
                    "placement_bid_regime_expected_edge_ewm_bps": values[50],
                    "placement_ask_regime_expected_edge_ewm_bps": values[51],
                }
            )
    return out


def write_aggregate(summaries: list[dict]) -> Path:
    tag = RESULT_TAG or f"edge_scored_{EXCHANGE_MODEL}"
    out = RESULT_DIR / f"bitmex_xbtusdt_edge_scored_maker_{tag}.aggregate.csv"
    fields = [
        "date",
        "total_pnl_usdt",
        "gross_pnl_before_fee_usdt",
        "maker_rebate_usdt",
        "fills",
        "buy_fills",
        "sell_fills",
        "filled_base_btc",
        "max_position_contracts_seen",
        "reduce_only_after_soft_position",
        "dynamic_quote",
        "dynamic_quote_expected_edge_mult",
        "dynamic_quote_edge_if_filled_mult",
        "dynamic_quote_max_tighten_bps",
        "dynamic_quote_fill_prob_widen_mult",
        "dynamic_quote_fill_prob_baseline",
        "dynamic_quote_max_widen_bps",
        "min_placement_expected_margin_bps",
        "min_placement_edge_if_filled_margin_bps",
        "daily_loss_limit_usdt",
        "daily_fill_limit",
        "regime_expected_edge_gate",
        "regime_min_bid_expected_edge_bps",
        "regime_min_ask_expected_edge_bps",
        "score_gate_bid",
        "score_gate_ask",
        "risk_halt_gate_bid",
        "risk_halt_gate_ask",
        "regime_gate_bid",
        "regime_gate_ask",
        "soft_position_gate_bid",
        "soft_position_gate_ask",
        "score_pass_bid",
        "score_pass_ask",
        "avg_gate_bid_expected_edge_bps",
        "avg_gate_ask_expected_edge_bps",
        "avg_gate_bid_edge_if_filled_bps",
        "avg_gate_ask_edge_if_filled_bps",
        "avg_gate_bid_fill_prob",
        "avg_gate_ask_fill_prob",
        "avg_quote_bid_half_spread_bps",
        "avg_quote_ask_half_spread_bps",
        "filter_cancel_bid",
        "filter_cancel_ask",
        "avg_spread_capture_usdt_per_btc",
        "avg_pred_bid_expected_edge_bps",
        "avg_pred_ask_expected_edge_bps",
        "avg_pred_bid_fill_prob",
        "avg_pred_ask_fill_prob",
        "final_regime_bid_expected_edge_ewm_bps",
        "final_regime_ask_expected_edge_ewm_bps",
        "avg_regime_bid_expected_edge_ewm_bps",
        "avg_regime_ask_expected_edge_ewm_bps",
        "force_close_pnl_usdt",
        "final_position_contracts",
    ]
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in fields})
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest BitMEX XBTUSDT edge-scored maker.")
    parser.add_argument("--dates", nargs="+", default=["20260516", "20260517", "20260518"])
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--buffer-rows", type=int, default=None)
    parser.add_argument("--result-tag", default="")
    parser.add_argument("--exchange-model", choices=("no_partial", "strict_no_partial", "partial"), default=EXCHANGE_MODEL)
    parser.add_argument("--order-qty-contracts", type=float, default=ORDER_QTY)
    parser.add_argument("--maker-fee-rate", type=float, default=MAKER_FEE_RATE)
    parser.add_argument("--taker-fee-rate", type=float, default=TAKER_FEE_RATE)
    parser.add_argument("--base-half-spread-bps", type=float, default=BASE_HALF_SPREAD_BPS)
    parser.add_argument("--dynamic-quote", action="store_true")
    parser.add_argument("--dynamic-quote-expected-edge-mult", type=float, default=DYNAMIC_QUOTE_EXPECTED_EDGE_MULT)
    parser.add_argument("--dynamic-quote-edge-if-filled-mult", type=float, default=DYNAMIC_QUOTE_EDGE_IF_FILLED_MULT)
    parser.add_argument("--dynamic-quote-max-tighten-bps", type=float, default=DYNAMIC_QUOTE_MAX_TIGHTEN_BPS)
    parser.add_argument("--dynamic-quote-fill-prob-widen-mult", type=float, default=DYNAMIC_QUOTE_FILL_PROB_WIDEN_MULT)
    parser.add_argument("--dynamic-quote-fill-prob-baseline", type=float, default=DYNAMIC_QUOTE_FILL_PROB_BASELINE)
    parser.add_argument("--dynamic-quote-max-widen-bps", type=float, default=DYNAMIC_QUOTE_MAX_WIDEN_BPS)
    parser.add_argument("--max-position-contracts", type=float, default=MAX_POSITION_CONTRACTS)
    parser.add_argument("--soft-position-contracts", type=float, default=SOFT_POSITION_CONTRACTS)
    parser.add_argument("--order-ttl-ms", type=float, default=ORDER_TTL_NS / 1_000_000.0)
    parser.add_argument("--rest-min-interval-ms", type=float, default=REST_MIN_INTERVAL_NS / 1_000_000.0)
    parser.add_argument("--model-dir", default="results/factor_research")
    parser.add_argument("--model-tag", default="")
    parser.add_argument("--model-file", default="", help="Use one explicit model file for all dates.")
    parser.add_argument("--expected-edge-threshold-bps", type=float, default=0.02)
    parser.add_argument("--edge-if-filled-threshold-bps", type=float, default=-999.0)
    parser.add_argument("--fill-prob-threshold", type=float, default=0.02)
    parser.add_argument("--placement-expected-margin-bps", type=float, default=MIN_PLACEMENT_EXPECTED_MARGIN_BPS)
    parser.add_argument(
        "--placement-edge-if-filled-margin-bps",
        type=float,
        default=MIN_PLACEMENT_EDGE_IF_FILLED_MARGIN_BPS,
    )
    parser.add_argument("--intraday-percentile-gate", action="store_true")
    parser.add_argument("--expected-edge-percentile", type=float, default=0.0)
    parser.add_argument("--edge-if-filled-percentile", type=float, default=0.0)
    parser.add_argument("--fill-prob-percentile", type=float, default=0.0)
    parser.add_argument("--reduce-only-after-soft-position", action="store_true")
    parser.add_argument("--daily-loss-limit-usdt", type=float, default=DAILY_LOSS_LIMIT_USDT)
    parser.add_argument("--daily-fill-limit", type=int, default=DAILY_FILL_LIMIT)
    parser.add_argument("--regime-expected-edge-gate", action="store_true")
    parser.add_argument("--regime-expected-edge-warmup-samples", type=int, default=REGIME_EXPECTED_EDGE_WARMUP_SAMPLES)
    parser.add_argument("--regime-expected-edge-ewm-alpha", type=float, default=REGIME_EXPECTED_EDGE_EWM_ALPHA)
    parser.add_argument("--regime-min-bid-expected-edge-bps", type=float, default=REGIME_MIN_BID_EXPECTED_EDGE_BPS)
    parser.add_argument("--regime-min-ask-expected-edge-bps", type=float, default=REGIME_MIN_ASK_EXPECTED_EDGE_BPS)
    parser.add_argument("--percentile-warmup-samples", type=int, default=PERCENTILE_WARMUP_SAMPLES)
    parser.add_argument("--percentile-update-interval-ms", type=float, default=PERCENTILE_UPDATE_INTERVAL_NS / 1_000_000.0)
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    global RESULT_TAG, EXCHANGE_MODEL, ORDER_QTY, MAKER_FEE_RATE, TAKER_FEE_RATE
    global BASE_HALF_SPREAD_BPS, MAX_POSITION_CONTRACTS, SOFT_POSITION_CONTRACTS
    global USE_DYNAMIC_QUOTE, DYNAMIC_QUOTE_EXPECTED_EDGE_MULT, DYNAMIC_QUOTE_EDGE_IF_FILLED_MULT
    global DYNAMIC_QUOTE_MAX_TIGHTEN_BPS, DYNAMIC_QUOTE_FILL_PROB_WIDEN_MULT
    global DYNAMIC_QUOTE_FILL_PROB_BASELINE, DYNAMIC_QUOTE_MAX_WIDEN_BPS
    global ORDER_TTL_NS, REST_MIN_INTERVAL_NS
    global MIN_EXPECTED_EDGE_BPS, MIN_EDGE_IF_FILLED_BPS, MIN_FILL_PROB
    global MIN_PLACEMENT_EXPECTED_MARGIN_BPS, MIN_PLACEMENT_EDGE_IF_FILLED_MARGIN_BPS
    global USE_INTRADAY_PERCENTILE_GATE, EXPECTED_EDGE_PERCENTILE, EDGE_IF_FILLED_PERCENTILE, FILL_PROB_PERCENTILE
    global REDUCE_ONLY_AFTER_SOFT_POSITION, DAILY_LOSS_LIMIT_USDT, DAILY_FILL_LIMIT
    global USE_REGIME_EXPECTED_EDGE_GATE, REGIME_EXPECTED_EDGE_WARMUP_SAMPLES, REGIME_EXPECTED_EDGE_EWM_ALPHA
    global REGIME_MIN_BID_EXPECTED_EDGE_BPS, REGIME_MIN_ASK_EXPECTED_EDGE_BPS
    global PERCENTILE_WARMUP_SAMPLES, PERCENTILE_UPDATE_INTERVAL_NS

    RESULT_TAG = args.result_tag
    EXCHANGE_MODEL = args.exchange_model
    ORDER_QTY = args.order_qty_contracts
    MAKER_FEE_RATE = args.maker_fee_rate
    TAKER_FEE_RATE = args.taker_fee_rate
    BASE_HALF_SPREAD_BPS = args.base_half_spread_bps
    USE_DYNAMIC_QUOTE = args.dynamic_quote
    DYNAMIC_QUOTE_EXPECTED_EDGE_MULT = args.dynamic_quote_expected_edge_mult
    DYNAMIC_QUOTE_EDGE_IF_FILLED_MULT = args.dynamic_quote_edge_if_filled_mult
    DYNAMIC_QUOTE_MAX_TIGHTEN_BPS = args.dynamic_quote_max_tighten_bps
    DYNAMIC_QUOTE_FILL_PROB_WIDEN_MULT = args.dynamic_quote_fill_prob_widen_mult
    DYNAMIC_QUOTE_FILL_PROB_BASELINE = args.dynamic_quote_fill_prob_baseline
    DYNAMIC_QUOTE_MAX_WIDEN_BPS = args.dynamic_quote_max_widen_bps
    MAX_POSITION_CONTRACTS = args.max_position_contracts
    SOFT_POSITION_CONTRACTS = args.soft_position_contracts
    ORDER_TTL_NS = int(args.order_ttl_ms * 1_000_000)
    REST_MIN_INTERVAL_NS = int(args.rest_min_interval_ms * 1_000_000)
    MIN_EXPECTED_EDGE_BPS = args.expected_edge_threshold_bps
    MIN_EDGE_IF_FILLED_BPS = args.edge_if_filled_threshold_bps
    MIN_FILL_PROB = args.fill_prob_threshold
    MIN_PLACEMENT_EXPECTED_MARGIN_BPS = args.placement_expected_margin_bps
    MIN_PLACEMENT_EDGE_IF_FILLED_MARGIN_BPS = args.placement_edge_if_filled_margin_bps
    USE_INTRADAY_PERCENTILE_GATE = args.intraday_percentile_gate
    EXPECTED_EDGE_PERCENTILE = args.expected_edge_percentile
    EDGE_IF_FILLED_PERCENTILE = args.edge_if_filled_percentile
    FILL_PROB_PERCENTILE = args.fill_prob_percentile
    REDUCE_ONLY_AFTER_SOFT_POSITION = args.reduce_only_after_soft_position
    DAILY_LOSS_LIMIT_USDT = args.daily_loss_limit_usdt
    DAILY_FILL_LIMIT = args.daily_fill_limit
    USE_REGIME_EXPECTED_EDGE_GATE = args.regime_expected_edge_gate
    REGIME_EXPECTED_EDGE_WARMUP_SAMPLES = args.regime_expected_edge_warmup_samples
    REGIME_EXPECTED_EDGE_EWM_ALPHA = args.regime_expected_edge_ewm_alpha
    REGIME_MIN_BID_EXPECTED_EDGE_BPS = args.regime_min_bid_expected_edge_bps
    REGIME_MIN_ASK_EXPECTED_EDGE_BPS = args.regime_min_ask_expected_edge_bps
    PERCENTILE_WARMUP_SAMPLES = args.percentile_warmup_samples
    PERCENTILE_UPDATE_INTERVAL_NS = int(args.percentile_update_interval_ms * 1_000_000)


def main() -> None:
    args = parse_args()
    apply_args(args)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "score_gate "
        f"expected_edge>={MIN_EXPECTED_EDGE_BPS}bps "
        f"edge_if_filled>={MIN_EDGE_IF_FILLED_BPS}bps "
        f"fill_prob>={MIN_FILL_PROB} "
        f"placement_margin>=({MIN_PLACEMENT_EXPECTED_MARGIN_BPS},{MIN_PLACEMENT_EDGE_IF_FILLED_MARGIN_BPS})bps "
        f"intraday_percentile={USE_INTRADAY_PERCENTILE_GATE} "
        f"pcts=({EXPECTED_EDGE_PERCENTILE},{EDGE_IF_FILLED_PERCENTILE},{FILL_PROB_PERCENTILE}) "
        f"reduce_only_after_soft={REDUCE_ONLY_AFTER_SOFT_POSITION} "
        f"daily_loss_limit={DAILY_LOSS_LIMIT_USDT} "
        f"daily_fill_limit={DAILY_FILL_LIMIT} "
        f"regime_gate={USE_REGIME_EXPECTED_EDGE_GATE} "
        f"regime_min=({REGIME_MIN_BID_EXPECTED_EDGE_BPS},{REGIME_MIN_ASK_EXPECTED_EDGE_BPS}) "
        f"regime_alpha={REGIME_EXPECTED_EDGE_EWM_ALPHA} "
        f"dynamic_quote={USE_DYNAMIC_QUOTE} "
        f"dynamic_mult=({DYNAMIC_QUOTE_EXPECTED_EDGE_MULT},{DYNAMIC_QUOTE_EDGE_IF_FILLED_MULT}) "
        f"dynamic_max_tighten={DYNAMIC_QUOTE_MAX_TIGHTEN_BPS} "
        f"dynamic_fill_widen=({DYNAMIC_QUOTE_FILL_PROB_WIDEN_MULT},{DYNAMIC_QUOTE_FILL_PROB_BASELINE})"
    )

    key = None if args.skip_download else tardis_key()
    summaries = []
    outputs = []
    for yyyymmdd in args.dates:
        model_path = find_model_file(Path(args.model_dir), args.model_tag, yyyymmdd, args.model_file)
        print(f"model {yyyymmdd}: {model_path}")
        if not args.skip_download:
            download_file(BITMEX_EXCHANGE, "trades", BITMEX_SYMBOL, yyyymmdd, key)
            download_file(BITMEX_EXCHANGE, "incremental_book_L2", BITMEX_SYMBOL, yyyymmdd, key)
        bitmex_npz = convert_bitmex(BITMEX_SYMBOL, yyyymmdd, args.buffer_rows)
        result = run_backtest(bitmex_npz, yyyymmdd, model_path)
        outputs.append(result)
        summary_path = result.with_suffix(".summary.json")
        summaries.append(json.loads(summary_path.read_text()))

    aggregate = write_aggregate(summaries)
    total_pnl = sum(item["total_pnl_usdt"] for item in summaries)
    total_gross = sum(item["gross_pnl_before_fee_usdt"] for item in summaries)
    total_rebate = sum(item["maker_rebate_usdt"] for item in summaries)
    total_fills = sum(item["fills"] for item in summaries)
    print(
        f"aggregate={aggregate} total_pnl={total_pnl:.6f} gross={total_gross:.6f} "
        f"rebate={total_rebate:.6f} fills={total_fills}"
    )
    print("all_results=" + ",".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
