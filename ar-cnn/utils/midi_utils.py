# The MIT-Zero License

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
from music21 import midi
import pypianoroll
from pypianoroll import Multitrack
from texttable import Texttable

from constants import Constants

def process_midi(midi_file, beat_resolution, program=0):
    '''Takes path to an input midi file and parses it to pianoroll
    :param input_midi: Path to midi file
    :param beat_resolution
    :return: parsed painoroll
    '''
    multi_track = pypianoroll.Multitrack(beat_resolution=beat_resolution)
    try:
        multi_track.parse_midi(midi_file,
                               skip_empty_tracks=False,
                               #collect_onsets_only=True,
                               algorithm='custom',
                               first_beat_time=0)
    except:
        print("midi file: {} is invalid. Ignoring during preprocessing".format(
            midi_file))
        pass
        
    # Convert the PianoRoll to binary ignoring the values of velocities
    
    multi_track.binarize()
    track_indices = [] 		# to merge all the voices
    track_indices_drum = [] # to merge drums
    for index,track in enumerate(multi_track.tracks):
        if track.is_drum:
            track_indices_drum.append(index)
        else:
            track_indices.append(index)
           
    if len(track_indices) == 0:
        print("midi file: {} is empty. Ignoring during preprocessing".format(midi_file))
        pass
          
    if len(track_indices_drum) == 0:
        print("midi file: {} has no drum. Ignoring during preprocessing".format(midi_file))
        pass
        
    multi_track.merge_tracks(track_indices=track_indices,
                             mode='any',
                             program=program,
                             remove_merged=False)
         
    pianoroll = multi_track.tracks[0].pianoroll
    drums = np.zeros((pianoroll.shape[0], 128), bool)
                       
    if len(track_indices_drum) > 0:
        multi_track.merge_tracks(track_indices=track_indices_drum,
                                 mode='any',
                                 is_drum=True,
                                 remove_merged=False)
        drums = multi_track.tracks[1].pianoroll         
    
    return pianoroll,drums


def process_pianoroll(pianoroll, drums,
                      total_time_steps_shifted_per_sample,
                      total_timesteps_per_nbars):
    '''Takes path to an input midi file and parses it to pianoroll
    :param pianoroll: pianoroll obtained after parsing midi file
    :param time_steps_shifted_per_sample: number of bars to be shifted in timesteps
    param timesteps_per_nbars: total number of timesteps to be included in processed pianoroll
    :return: parsed painoroll sections
    '''

    if Constants.split_into_two_voices:
        time_steps_shifted_per_sample = total_time_steps_shifted_per_sample // 2
        timesteps_per_nbars = total_timesteps_per_nbars // 2
    else:
        time_steps_shifted_per_sample = total_time_steps_shifted_per_sample
        timesteps_per_nbars = total_timesteps_per_nbars
        # instead squeeze drums at the end of the pianoroll
        drum_subset = np.take(drums, axis=1, indices=Constants.drums)
        drum_end_fill = np.zeros((pianoroll.shape[0], 128 - Constants.voices_maximum - drum_subset.shape[1]), bool)
        np.resize(pianoroll, [pianoroll.shape[0],Constants.voices_maximum])
        np.concatenate((pianoroll, drum_subset), axis=1)
        np.concatenate((pianoroll, drum_end_fill), axis=1)

    pianoroll_sections = []
    truncated_pianoroll_length = pianoroll.shape[0] - (pianoroll.shape[0] %
                                                       timesteps_per_nbars)
                                                       
    for i in range(0, truncated_pianoroll_length - timesteps_per_nbars + 1,
                   time_steps_shifted_per_sample):
					
        if Constants.split_into_two_voices:
            pianoroll_section = pianoroll[i:i + timesteps_per_nbars, :]
            drum_section = drums[i:i + timesteps_per_nbars, :]					   
            section = np.concatenate((pianoroll_section, drum_section))	
        else:	
            section = pianoroll[i:i + timesteps_per_nbars, :]
        
        pianoroll_sections.append(section)
        
    return pianoroll_sections

def play_midi(input_midi):
    '''Takes path to an input and plays the midi file in the notebook cell
    :param input_midi: Path to midi file
    :return:
    '''
    midi_object = midi.MidiFile()
    midi_object.open(input_midi)
    midi_object.read()
    midi_object.close()
    show_midi = midi.translate.midiFileToStream(midi_object)
    show_midi.show('midi')


def plot_pianoroll(pianoroll, beat_resolution, fig_name=None):
    '''Plot a Single Track Pianoroll image
    :param pianoroll: Pianoroll tensor of shape time_steps * pitches
    :return:
    '''
    pypianoroll.plot_track(pypianoroll.Track(pianoroll=pianoroll), fig_name,
                           beat_resolution)


def get_music_metrics(input_midi, beat_resolution, track=0):
    """Takes a midifile as an input and Returns the following metrics
    :param input_midi: Path to midi file
    :param beat_resolution:
    :param trac: Instrument number in the multi track midi file
    :return: The following metrics are returned
    1.) n_pitch_classes_used is the unique pitch classes used in a pianoroll.
    2.) polyphonic_rate ie ratio of the number of time steps where the number of pitches
        being played is larger than `threshold` to the total number of time steps in
        a pianoroll
    3.) in_scale_rate is the ratio of the number of nonzero entries that lie in a specific scale
        to the total number of nonzero entries in a pianoroll.
    4.) n_pitches_used is the the number of unique pitches used in a pianoroll."""

    midi_data = Multitrack(input_midi, beat_resolution)
    piano_roll = midi_data.tracks[track].pianoroll
    n_pitch_classes_used = pypianoroll.metrics.n_pitch_classes_used(piano_roll)
    polyphonic_rate = pypianoroll.metrics.polyphonic_rate(piano_roll)
    in_scale_rate = pypianoroll.metrics.in_scale_rate(piano_roll)
    n_pitches_used = pypianoroll.metrics.n_pitches_used(piano_roll)
    metrics = [
        n_pitch_classes_used, polyphonic_rate, n_pitches_used, in_scale_rate
    ]
    metrics_table = [[
        "n_pitch_classes_used", "in_scale_rate", "polyphonic_rate",
        "n_pitches_used"
    ], metrics]
    table = Texttable()
    table.add_rows(metrics_table)
    print(table.draw())
