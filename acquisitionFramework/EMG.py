from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from mindrove.data_filter import DataFilter, DetrendOperations

EMG_SFREQ = 500
EMG_N_CHANNEL = 8

class MindRoveEMG:
    def __init__(self):
        BoardShim.enable_dev_board_logger()
        params = MindRoveInputParams()
        self.board_shim = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)
        self.board_shim.prepare_session()
        self.board_shim.start_stream()
        board_id = self.board_shim.get_board_id()
        self.exg_channels = BoardShim.get_exg_channels(board_id)

    def getEMG(self):
        data = self.board_shim.get_board_data()
        try:
            if len(data[0]) >= 1:
                for count, channel in enumerate(self.exg_channels):
                    DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
        except Exception as e:
            print("could not detrend data. Continue to run")

        return data[:8]

    def clearBuffer(self):
        self.board_shim.get_board_data()
