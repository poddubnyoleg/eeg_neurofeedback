# import muselsl.viewer_v1 as v
# v.view(5., 100, 0.2, "15x6", 1)


from pylsl import StreamInlet, resolve_byprop

streams = resolve_byprop('type', 'EEG', timeout=5)
inlet = StreamInlet(streams[0], max_chunklen=12)
eeg_time_correction = inlet.time_correction()


print 'here'