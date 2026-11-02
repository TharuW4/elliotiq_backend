from .MonoWave import MonoWaveUp, MonoWaveDown
from .WaveOptions import WaveOptionsGenerator5, WaveOptionsGenerator3
from .WaveCycle import WaveCycle
from .WavePattern import WavePattern
from .WaveRules import Impulse, Correction, TDWave, BearishCorrection
import numpy as np
import pandas as pd


class WaveAnalyzer:
    """
    Find impulse or corrective waves for given dataframe
    """
    def __init__(self,
                 df: pd.DataFrame,
                 verbose: bool = False):

        self.df = df
        self.lows = np.array(list(self.df['Low']))
        self.highs = np.array(list(self.df['High']))
        self.dates = np.array(list(self.df['Date']))
        self.verbose = verbose

        self.impulse_rules = list()
        self.correction_rules = list()

        self.__waveoptions_up: WaveOptionsGenerator5
        self.__waveoptions_down: WaveOptionsGenerator3

        self.set_combinatorial_limits()

    def get_absolute_low(self):
        """
        find the absolute low in the dataframe. Can be used to start the wave analysis from this low.
        :return:
        """
        return np.min(self.lows)

    def set_combinatorial_limits(self, n_up: int = 10, n_down: int = 10):
        """
        Change the limit to skip min / maxima for the WaveOptionsGenerators, e.g. go up to [n_up, n_up, ...] for the
        WaveOptions

        :param n_up:
        :param n_down:
        :return:
        """
        self.__waveoptions_up = WaveOptionsGenerator5(n_up)
        self.__waveoptions_down = WaveOptionsGenerator3(n_down)

    # ==========================================================
    # NEW: Internal helpers for bearish impulse validation
    # ==========================================================

    def _range_max(self, arr, idx_a, idx_b):
        try:
            if idx_a is None or idx_b is None:
                return None

            start = max(0, min(int(idx_a), int(idx_b)))
            end = min(len(arr), max(int(idx_a), int(idx_b)) + 1)

            if end <= start:
                return None

            return float(np.max(arr[start:end]))
        except Exception:
            return None

    def _wave_length(self, wave):
        try:
            return abs(float(wave.high) - float(wave.low))
        except Exception:
            return None

    def find_impulsive_wave(self,
                            idx_start: int,
                            wave_config: list = None):
        """
        Tries to find 5 consecutive waves (up, down, up, down, up) to build an impulsive 12345 wave

        :param idx_start: index in dataframe to start from
        :param wave_config: WaveOptions
        :return: list of the 5 MonoWaves in case they are found.

                False otherwise
        """

        if wave_config is None:
            wave_config = [0, 0, 0, 0, 0]

        wave1 = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=idx_start, skip=wave_config[0])
        wave1.label = '1'
        wave1_end = wave1.idx_end
        if wave1_end is None:
            if self.verbose: print("Wave 1 has no End in Data")
            return False

        wave2 = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=wave1_end, skip=wave_config[1])
        wave2.label = '2'
        wave2_end = wave2.idx_end
        if wave2_end is None:
            if self.verbose: print("Wave 2 has no End in Data")
            return False

        wave3 = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=wave2_end, skip=wave_config[2])
        wave3.label = '3'
        wave3_end = wave3.idx_end
        if wave3_end is None:
            if self.verbose: print("Wave 3 has no End in Data")
            return False

        wave4 = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=wave3_end, skip=wave_config[3])
        wave4.label = '4'
        wave4_end = wave4.idx_end

        if wave4_end is None:
            if self.verbose: print("Wave 4 has no End in Data")
            return False

        if wave2.low > np.min(self.lows[wave2.low_idx:wave4.low_idx]):
            return False

        wave5 = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=wave4_end, skip=wave_config[4])
        wave5.label = '5'
        wave5_end = wave5.idx_end
        if wave5_end is None:
            if self.verbose: print("Wave 5 has no End in Data")
            return False

        if self.lows[wave4.low_idx:wave5.high_idx].any() and wave4.low > np.min(self.lows[wave4.low_idx:wave5.high_idx]):
            if self.verbose: print('Low of Wave 4 higher than a low between Wave 4 and Wave 5')
            return False

        # --- NEW: Wave 3 Extension Check (Bullish) ---
        try:
            len1 = abs(float(wave1.high) - float(wave1.low))
            len3 = abs(float(wave3.high) - float(wave3.low))
            if len3 > (len1 * 1.618):
                setattr(wave3, 'is_extended', True) 
            else:
                setattr(wave3, 'is_extended', False)
        except Exception:
            setattr(wave3, 'is_extended', False)

        return [wave1, wave2, wave3, wave4, wave5]

    # ==========================================================
    # NEW: Bearish impulse validation
    # ==========================================================

    def _is_bearish_impulse_structurally_valid(self, wave1, wave2, wave3, wave4, wave5):
        """
        Validate bearish impulse structure:
        Wave 1: down
        Wave 2: up
        Wave 3: down
        Wave 4: up
        Wave 5: down

        This is intentionally medium-strict to avoid sending false bearish Elliott waves to AI.
        """
        try:
            required = [
                wave1.high, wave1.low, wave1.high_idx, wave1.low_idx,
                wave2.high, wave2.low, wave2.high_idx, wave2.low_idx,
                wave3.high, wave3.low, wave3.high_idx, wave3.low_idx,
                wave4.high, wave4.low, wave4.high_idx, wave4.low_idx,
                wave5.high, wave5.low, wave5.high_idx, wave5.low_idx,
            ]

            if any(v is None for v in required):
                return False

            # Wave 2 should not retrace above Wave 1 start high.
            if float(wave2.high) >= float(wave1.high):
                if self.verbose:
                    print("Bearish impulse invalid: Wave 2 retraced above Wave 1 start.")
                return False

            # Wave 3 should break below Wave 1 low.
            if float(wave3.low) >= float(wave1.low):
                if self.verbose:
                    print("Bearish impulse invalid: Wave 3 did not break below Wave 1 low.")
                return False

            # Wave 4 should not overlap Wave 1 territory.
            # For bearish impulse, Wave 4 high should stay below Wave 1 low.
            if float(wave4.high) >= float(wave1.low):
                if self.verbose:
                    print("Bearish impulse invalid: Wave 4 overlaps Wave 1 territory.")
                return False

            # Wave 5 should continue below Wave 3 low.
            # This rejects truncated fifth waves for safety.
            # --- UPDATED: Commented out to allow Truncated Wave 5 (Reversals) ---
            # if float(wave5.low) >= float(wave3.low):
            #     if self.verbose:
            #         print("Bearish impulse invalid: Wave 5 did not break below Wave 3 low.")
            #     return False

            # Wave 3 should not be the shortest among waves 1, 3, 5.
            len1 = self._wave_length(wave1)
            len3 = self._wave_length(wave3)
            len5 = self._wave_length(wave5)

            if len1 is None or len3 is None or len5 is None:
                return False

            if len3 < min(len1, len5):
                if self.verbose:
                    print("Bearish impulse invalid: Wave 3 is shortest.")
                return False

            # Between Wave 2 and Wave 4, price should not exceed Wave 2 high.
            highest_between_2_4 = self._range_max(self.highs, wave2.high_idx, wave4.high_idx)
            if highest_between_2_4 is not None and highest_between_2_4 > float(wave2.high):
                if self.verbose:
                    print("Bearish impulse invalid: price exceeded Wave 2 high before/at Wave 4.")
                return False

            # Between Wave 4 and Wave 5, price should not exceed Wave 4 high.
            highest_between_4_5 = self._range_max(self.highs, wave4.high_idx, wave5.low_idx)
            if highest_between_4_5 is not None and highest_between_4_5 > float(wave4.high):
                if self.verbose:
                    print("Bearish impulse invalid: price exceeded Wave 4 high before Wave 5 completed.")
                return False

            return True

        except Exception as e:
            if self.verbose:
                print(f"Bearish impulse validation error: {e}")
            return False

    # ==========================================================
    # NEW: Bearish impulse finder
    # ==========================================================

    def find_impulsive_wave_down(self, idx_start: int, wave_config: list = None):
        """
        Find bearish impulse:
        down, up, down, up, down

        This method is separate from find_impulsive_wave()
        so existing bullish logic stays unchanged.
        """
        if wave_config is None:
            wave_config = [0, 0, 0, 0, 0]

        wave1 = MonoWaveDown(
            lows=self.lows,
            highs=self.highs,
            dates=self.dates,
            idx_start=idx_start,
            skip=wave_config[0]
        )
        wave1.label = "1"
        wave1_end = wave1.idx_end
        if wave1_end is None:
            if self.verbose:
                print("Bearish Wave 1 has no End in Data")
            return False

        wave2 = MonoWaveUp(
            lows=self.lows,
            highs=self.highs,
            dates=self.dates,
            idx_start=wave1_end,
            skip=wave_config[1]
        )
        wave2.label = "2"
        wave2_end = wave2.idx_end
        if wave2_end is None:
            if self.verbose:
                print("Bearish Wave 2 has no End in Data")
            return False

        wave3 = MonoWaveDown(
            lows=self.lows,
            highs=self.highs,
            dates=self.dates,
            idx_start=wave2_end,
            skip=wave_config[2]
        )
        wave3.label = "3"
        wave3_end = wave3.idx_end
        if wave3_end is None:
            if self.verbose:
                print("Bearish Wave 3 has no End in Data")
            return False

        wave4 = MonoWaveUp(
            lows=self.lows,
            highs=self.highs,
            dates=self.dates,
            idx_start=wave3_end,
            skip=wave_config[3]
        )
        wave4.label = "4"
        wave4_end = wave4.idx_end
        if wave4_end is None:
            if self.verbose:
                print("Bearish Wave 4 has no End in Data")
            return False

        wave5 = MonoWaveDown(
            lows=self.lows,
            highs=self.highs,
            dates=self.dates,
            idx_start=wave4_end,
            skip=wave_config[4]
        )
        wave5_end = wave5.idx_end
        if wave5_end is None:
            if self.verbose:
                print("Bearish Wave 5 has no End in Data")
            return False

        if not self._is_bearish_impulse_structurally_valid(wave1, wave2, wave3, wave4, wave5):
            return False

        # --- NEW: Wave 3 Extension Check (Bearish) ---
        try:
            len1 = abs(float(wave1.low) - float(wave1.high))
            len3 = abs(float(wave3.low) - float(wave3.high))
            if len3 > (len1 * 1.618):
                setattr(wave3, 'is_extended', True)
            else:
                setattr(wave3, 'is_extended', False)
        except Exception:
            setattr(wave3, 'is_extended', False)

        return [wave1, wave2, wave3, wave4, wave5]

    def find_corrective_wave(self,
                             idx_start: int,
                             wave_config: list = None):
        """

        Tries to find a corrective movement (ABC)
        :param idx_start:
        :param wave_config:
        :return: a list of 3 MonoWaves (down, up, down) otherwise False

        """
        if wave_config is None:
            wave_config = [0, 0, 0]

        waveA = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=idx_start, skip=wave_config[0])
        waveA.label = 'A'
        waveA_end = waveA.idx_end
        if waveA_end is None:
            return False

        waveB = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=waveA_end, skip=wave_config[1])
        waveB.label = 'B'
        waveB_end = waveB.idx_end
        if waveB_end is None:
            return False

        waveC = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=waveB_end, skip=wave_config[2])
        waveC.label = 'C'
        waveC_end = waveC.idx_end
        if waveC_end is None:
            return False

        return [waveA, waveB, waveC]

    def find_corrective_wave_up(self,
                             idx_start: int,
                             wave_config: list = None):
        """
        Tries to find a bearish corrective movement (A=Up, B=Down, C=Up)
        after a bearish impulse.
        """
        if wave_config is None:
            wave_config = [0, 0, 0]

        waveA = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=idx_start, skip=wave_config[0])
        waveA.label = 'A'
        waveA_end = waveA.idx_end
        if waveA_end is None:
            return False

        waveB = MonoWaveDown(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=waveA_end, skip=wave_config[1])
        waveB.label = 'B'
        waveB_end = waveB.idx_end
        if waveB_end is None:
            return False

        waveC = MonoWaveUp(lows=self.lows, highs=self.highs, dates=self.dates, idx_start=waveB_end, skip=wave_config[2])
        waveC.label = 'C'
        waveC_end = waveC.idx_end
        if waveC_end is None:
            return False

        return [waveA, waveB, waveC]

    def find_td_wave(self, idx_start: int, wave_config: list = None):
        """
        Existing TD wave helper:
        up, down

        Fixed to use the same MonoWave constructor style as other methods.
        """
        if wave_config is None:
            wave_config = [0, 0]

        wave1 = MonoWaveUp(
            lows=self.lows,
            highs=self.highs,
            dates=self.dates,
            idx_start=idx_start,
            skip=wave_config[0]
        )
        wave1.label = '1'
        wave1_end = wave1.idx_end
        if wave1_end is None:
            if self.verbose:
                print("Wave 1 has no End in Data")
            return False

        wave2 = MonoWaveDown(
            lows=self.lows,
            highs=self.highs,
            dates=self.dates,
            idx_start=wave1_end,
            skip=wave_config[1]
        )
        wave2.label = '2'
        wave2_end = wave2.idx_end
        if wave2_end is None:
            if self.verbose:
                print("Wave 2 has no End in Data")
            return False

        return [wave1, wave2]

    def next_cycle(self, start_idx: int):

        impulse = Impulse('impulse')
        correction = Correction('correction')
        bearish_correction = BearishCorrection('bearish_correction')

        wave_cycles = set()

        # --- 1. BULLISH CYCLES (Impulse UP -> Correction DOWN) ---
        for new_option_impulse in self.__waveoptions_up.options_sorted:
            cycle_complete = False
            waves_up = self.find_impulsive_wave(idx_start=start_idx, wave_config=new_option_impulse.values)

            if waves_up:
                wavepattern_up = WavePattern(waves_up, verbose=False)
                if wavepattern_up.check_rule(impulse):
                    if self.verbose: print('Bullish Impulse found!', new_option_impulse.values)
                    end = waves_up[4].idx_end

                    for new_option_correction in self.__waveoptions_down.options_sorted:
                        waves = self.find_corrective_wave(idx_start=end, wave_config=new_option_correction.values)
                        if waves:
                            wavepattern = WavePattern(waves, verbose=False)
                            if wavepattern.check_rule(correction):
                                cycle_complete = True
                                wave_cycle = WaveCycle(wavepattern_up, wavepattern)
                                wave_cycles.add(wave_cycle)

                                if self.verbose and wave_cycle not in wave_cycles:
                                    print('Bullish Correction found!', new_option_correction.values)

                    if cycle_complete:
                        yield wave_cycle

        # --- 2. BEARISH CYCLES (Impulse DOWN -> Correction UP) ---
        # FIXED: Bearish impulse needs 5 waves, so use __waveoptions_up
        for new_option_impulse in self.__waveoptions_up.options_sorted:
            cycle_complete = False
            waves_down = self.find_impulsive_wave_down(idx_start=start_idx, wave_config=new_option_impulse.values)

            if waves_down:
                wavepattern_down = WavePattern(waves_down, verbose=False)
                if self.verbose: print('Bearish Impulse found!', new_option_impulse.values)
                end = waves_down[4].idx_end

                # FIXED: Bearish correction needs 3 waves, so use __waveoptions_down
                for new_option_correction in self.__waveoptions_down.options_sorted:
                    waves = self.find_corrective_wave_up(idx_start=end, wave_config=new_option_correction.values)
                    if waves:
                        wavepattern = WavePattern(waves, verbose=False)
                        if wavepattern.check_rule(bearish_correction): 
                            cycle_complete = True
                            wave_cycle = WaveCycle(wavepattern_down, wavepattern)
                            wave_cycles.add(wave_cycle)

                            if self.verbose and wave_cycle not in wave_cycles:
                                print('Bearish Correction found!', new_option_correction.values)

                    if cycle_complete:
                        yield wave_cycle

        return None