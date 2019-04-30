train_wave_split:
	make -C common/data_split split_per_earthquake

train_split:
	make -C common/data_split split_data
