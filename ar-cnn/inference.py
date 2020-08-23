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

import os
import logging
import pypianoroll
import keras
import numpy as np
from losses import Loss
from constants import Constants
import copy

from utils.midi_utils import print_midi_info, process_pianoroll, process_midi, plot_pianoroll

logger = logging.getLogger(__name__)


class Inference:
    def __init__(self, model=None):
        self.model = model

    def load_model(self, model_path):
        """
        Loads a trained keras model

        Parameters
        ----------
        model_path : string
            Full file path to the trained model

        Returns
        -------
        None
        """
        self.model = keras.models.load_model(model_path,
                                             custom_objects={
                                                 'built_in_softmax_kl_loss':
                                                 Loss.built_in_softmax_kl_loss
                                             },
                                             compile=False)

    @staticmethod
    def convert_tensors_to_midi(tensors, tempo, program, output_file_path):
        """
        Writes a pianoroll tensor to a midi file

        Parameters
        ----------
        tensor : 2d numpy array
            pianoroll to be converted to a midi
        tempo : float
            tempo to output
        output_file_path : str
            output midi file path

        Returns
        -------
        None
        """

        if len(tensors) == 0:
            return;

        tensor_size = tensors[0].shape[1]
        
        if Constants.split_into_two_voices:
            real_size = tensor_size // 2
        else:
            real_size = tensor_size

        pianoroll = np.zeros((0, 128), bool)
        drums = np.zeros((0, 128), bool)
            
        for tensor in tensors:			
            pianoroll_tensor = tensor[0:real_size,0:Constants.voices_maximum]
            drums_tensor = tensor[tensor_size-real_size:tensor_size,Constants.voices_maximum:]
            
            if not Constants.split_into_two_voices:
                # resize pianoroll tensor
                pianoroll_tensor = np.concatenate((pianoroll_tensor, np.zeros((real_size,128-Constants.voices_maximum), bool)), axis=1)
                # extract specific notes from drum tensor
                new_drums = np.zeros((tensor_size, 128), bool)
                for index,position in enumerate(Constants.drums):
                    new_drums[:,position] = drums_tensor[:,index]
                drums_tensor = new_drums
            
            pianoroll = np.concatenate((pianoroll, pianoroll_tensor))
            drums = np.concatenate((drums, drums_tensor))
                
        #plot_pianoroll(pianoroll,
        #    beat_resolution=Constants.beat_resolution,
        #    fig_name="output_pianoroll")
        #plot_pianoroll(drums,
        #    beat_resolution=Constants.beat_resolution,
        #    fig_name="output_drums")
            
        pianoroll_track = pypianoroll.Track(pianoroll=pianoroll, program=program)
        drums_track = pypianoroll.Track(pianoroll=drums, is_drum=True)
        
        multi_track = pypianoroll.Multitrack(
            tracks=[pianoroll_track, drums_track],
            tempo=tempo,
            beat_resolution=Constants.beat_resolution)
            
        #output_file_index = 0
        #while os.path.isfile(output_file_path.format(output_file_index)):
        #    output_file_index += 1
        #multi_track.write(output_file_path.format(output_file_index))
        multi_track.write(output_file_path)

    @staticmethod
    def get_indices(input_tensor, value):
        """
        Parameters
        ----------
        input_tensor : 2d numpy array
        value : int (either 1 or 0)

        Returns
        -------
        indices_with_value : 2d array of indices in the input_tensor where the pixel value equals value (1 or 0).
        """
        indices_with_value = np.argwhere(input_tensor.astype(np.bool_) == value)
        return set(map(tuple, indices_with_value))

    @staticmethod
    def get_softmax(input_tensor, temperature):
        """
        Gets the softmax of a tensor with temperature

        Parameters
        ----------
        input_tensor : numpy array
            original tensor (e.g. original predictions)
        temperature : int
            softmax temperature

        Returns
        -------
        tensor : numpy array
            softmax of input tensor with temperature
        """
        tensor = input_tensor / temperature
        tensor = np.exp(tensor)
        tensor = tensor / np.sum(tensor)
        return tensor

    @staticmethod
    def get_sampled_index(input_tensor):
        """
        Gets a randomly chosen index from the input tensor

        Parameters
        ----------
        input_tensor : numpy array
            original tensor
        Returns
        -------
        tensor : numpy array
            softmax of input tensor with temperature
        """

        sampled_index = np.random.choice(range(input_tensor.size),
                                         1,
                                         p=input_tensor.ravel())
        sampled_index = np.unravel_index(sampled_index, input_tensor.shape)
        return sampled_index

    def generate_composition(self, input_midi_path, output_midi_path, inference_params):
        """
        Generates a new composition based on an old midi

        Parameters
        ----------
        input_midi_path : str
            input midi path
        inference_params : json
            JSON with inference parameters

        Returns
        -------
        None
        """
        try:
            input_tensors = self.convert_midi_to_tensors(input_midi_path)
            output_tensors = []
            
            for input_tensor in input_tensors:
                output_tensor = self.sample_multiple(
                    input_tensor, inference_params['temperature'],
                    inference_params['maxPercentageOfInitialNotesRemoved'],
                    inference_params['maxNotesAdded'],
                    inference_params['samplingIterations'],    
                    inference_params['generateVoice'],                 
                    inference_params['generateDrums'])
                output_tensors.append(output_tensor)
                    
            self.convert_tensors_to_midi(output_tensors, Constants.tempo,
                                        Constants.program, output_midi_path)
        except Exception:
            logger.error("Unable to generate composition.")
            raise

    def convert_midi_to_tensors(self, input_midi_path):
        """
        Converts a midi to pianoroll tensor

        Parameters
        ----------
        input_midi_path : string
            Full file path to the input midi

        Returns
        -------
        2d numpy array
            2d tensor that is a pianoroll
        """

        print('process ' + input_midi_path + '...')
        #print_midi_info(input_midi_path)
        
        pianoroll, drums = process_midi(input_midi_path,
			beat_resolution=Constants.beat_resolution,
			program=Constants.program,
            ignore_warnings=True)

        #plot_pianoroll(pianoroll,
        #    beat_resolution=Constants.beat_resolution,
        #    fig_name="original_pianoroll")
        #plot_pianoroll(drums,
        #    beat_resolution=Constants.beat_resolution,
        #    fig_name="original_drums")
			
        tensors = process_pianoroll(pianoroll, drums,
			Constants.number_of_timesteps, Constants.number_of_timesteps)
            
        #combined_pianoroll = np.zeros((0, 128), bool)
        #for tensor in tensors:
        #    combined_pianoroll = np.concatenate((combined_pianoroll, tensor))        
        #plot_pianoroll(combined_pianoroll,
        #    beat_resolution=Constants.beat_resolution,
        #    fig_name="processed_pianoroll")
            
        for index, tensor in enumerate(tensors):
            tensor = np.expand_dims(tensor, axis=0)
            tensor = np.expand_dims(tensor, axis=3)
            tensors[index] = tensor

        return tensors

    def mask_not_allowed_notes(self, current_input_indices, output_tensor):
        """
        Masks notes in output tensor that cannot be added or removed

        Parameters
        ----------
        current_input_indices : 2d numpy array
          indices to be masked based on the current input that was fed to model
        output_tensor : 2d numpy array
          consists of probabilities that are predicted by the model

        Returns
        -------
        2d numpy array - output tensor with not allowed notes masked
        """

        if len(current_input_indices) != 0:
            output_tensor[tuple(np.asarray(list(current_input_indices)).T)] = 0
            if np.count_nonzero(output_tensor) != 0:
                output_tensor = output_tensor / np.sum(output_tensor)
        return output_tensor

    def sample_multiple(self, input_tensor, temperature,
                        max_removal_percentage, max_notes_to_add,
                        number_of_iterations, generate_voice, generate_drums):
        """
        Samples multiple times from an tensor.
        Returns the final output tensor after X number of iterations.

        Parameters
        ----------
        input_tensor : 2d numpy array
            original tensor (i.e. user input melody)
        temperature : float
            temperature to apply before softmax during inference
        max_removal_percentage : float
            maximum percentage of notes that can be removed from the original input
        max_notes_to_add : int
            maximum number of notes that can be added to the original input
        number_of_iterations : int
            number of iterations to sample from the model predictions
        generate_voice : bool
            if True, generate samples for the main voice
        generate_drums : bool
            if True, generate samples for the drum beats

        Returns
        -------
        2d numpy array
            output tensor (i.e. new composition)
        """

        max_original_notes_to_remove = int(
            max_removal_percentage * np.count_nonzero(input_tensor) / 100)
        notes_removed_count = 0
        notes_added_count = 0

        original_input_one_indices = self.get_indices(input_tensor, 1)
        original_input_zero_indices = self.get_indices(input_tensor, 0)

        current_input_one_indices = copy.deepcopy(original_input_one_indices)
        current_input_zero_indices = copy.deepcopy(original_input_zero_indices)

        if not generate_drums and not generate_voice:
            print("nothing to generate...")
        else:

            if not generate_drums:
                print("generate voice only...")

            if not generate_voice:            
                print("generate drums only...")

            for _ in range(number_of_iterations):
                    input_tensor, notes_removed_count, notes_added_count = self.sample_notes_from_model(
                        input_tensor, max_original_notes_to_remove, max_notes_to_add,
                        temperature, generate_voice, generate_drums,
                        notes_removed_count, notes_added_count,
                        original_input_one_indices, original_input_zero_indices,
                        current_input_zero_indices, current_input_one_indices)

        return input_tensor.reshape(Constants.number_of_timesteps,
                                    Constants.number_of_pitches)

    def sample_notes_from_model(self,
                                input_tensor,
                                max_original_notes_to_remove,
                                max_notes_to_add,
                                temperature,
                                generate_voice,
                                generate_drums,
                                notes_removed_count,
                                notes_added_count,
                                original_input_one_indices,
                                original_input_zero_indices,
                                current_input_zero_indices,
                                current_input_one_indices,
                                num_notes=1):
        """
        Generates a sample from the tensor and return a new tensor
        Modifies current_input_zero_indices, current_input_one_indices, and input_tensor

        Parameters
        ----------
        input_tensor : 2d numpy array
            input tensor to feed into the model
        max_original_notes_to_remove : int
            maximum number of notes to remove from the original input
        max_notes_to_add : int
            maximum number of notes that can be added to the original input
        temperature : float
            temperature to apply before softmax during inference
        generate_voice : bool
            if True, generate samples for the main voice
        generate_drums : bool
            if True, generate samples for the drum beats
        notes_removed_count : int
            number of original notes that have been removed from input
        notes_added_count : int
            number of new notes that have been added to the input
        original_input_one_indices : set of tuples
            indices which have value 1 in original input
        original_input_zero_indices : set of tuples
            indices which have value 0 in original input
        current_input_zero_indices : set of tuples
            indices which have value 0 and were not part of the original input
        current_input_one_indices : set of tuples
            indices which have value 1 and were part of the original input

        Returns
        -------
        input_tensor : 2d numpy array
            output after samping from the model prediction
        notes_removed_count : int
            updated number of original notes removed
        notes_added_count : int
            updated number of new notes added
        """

        if not generate_voice and not generate_drums:
            # nothing to generate, just return the input directly           
            return input_tensor, notes_removed_count, notes_added_count

        output_tensor = self.model.predict([input_tensor])

        # Apply temperature and softmax
        output_tensor = self.get_softmax(output_tensor, temperature)

        if notes_removed_count >= max_original_notes_to_remove:
            # Mask all pixels that both have a note and were once part of the original input
            output_tensor = self.mask_not_allowed_notes(current_input_one_indices, output_tensor)

        if notes_added_count > max_notes_to_add:
            # Mask all pixels that both do not have a note and were not once part of the original input
            output_tensor = self.mask_not_allowed_notes(current_input_zero_indices, output_tensor)

        output_tensor_voice = output_tensor.copy()
        output_tensor_drums = output_tensor.copy()

        output_tensor_voice[:,:,Constants.voices_maximum:,:].fill(0)
        output_tensor_drums[:,:,:Constants.voices_maximum,:].fill(0)
        
        if generate_voice:
            input_tensor, notes_removed_count, notes_added_count = self.sample_notes_from_model_tensor_output(
                input_tensor,
                output_tensor_voice,
                max_original_notes_to_remove,
                max_notes_to_add,
                notes_removed_count,
                notes_added_count,
                original_input_one_indices,
                original_input_zero_indices,
                current_input_zero_indices,
                current_input_one_indices)

        if generate_drums:            
            input_tensor, notes_removed_count, notes_added_count = self.sample_notes_from_model_tensor_output(
                input_tensor,
                output_tensor_drums,
                max_original_notes_to_remove,
                max_notes_to_add,
                notes_removed_count,
                notes_added_count,
                original_input_one_indices,
                original_input_zero_indices,
                current_input_zero_indices,
                current_input_one_indices)

        return input_tensor, notes_removed_count, notes_added_count
            
    def sample_notes_from_model_tensor_output(self,
                                              input_tensor,
                                              output_tensor,
                                              max_original_notes_to_remove,
                                              max_notes_to_add,
                                              notes_removed_count,
                                              notes_added_count,
                                              original_input_one_indices,
                                              original_input_zero_indices,
                                              current_input_zero_indices,
                                              current_input_one_indices):
        
        if np.count_nonzero(output_tensor) == 0:
            return input_tensor, notes_removed_count, notes_added_count

        output_tensor /= output_tensor.sum()

        sampled_index = self.get_sampled_index(output_tensor)
        sampled_index_transpose = tuple(np.array(sampled_index).T[0])

        if input_tensor[sampled_index]:
            # Check if the note being removed is from the original input
            if notes_removed_count < max_original_notes_to_remove and (
                sampled_index_transpose in original_input_one_indices):
                notes_removed_count += 1
                current_input_one_indices.remove(sampled_index_transpose)
            elif tuple(sampled_index_transpose) not in original_input_one_indices:
                notes_added_count -= 1
                current_input_zero_indices.add(sampled_index_transpose)
            input_tensor[sampled_index] = 0
        else:
            # Check if the note being added is not in original input
            if sampled_index_transpose not in original_input_one_indices:
                notes_added_count += 1
                current_input_zero_indices.remove(sampled_index_transpose)
            else:
                notes_removed_count -= 1
                current_input_one_indices.add(sampled_index_transpose)
            input_tensor[sampled_index] = 1
        input_tensor = input_tensor.astype(np.bool_)
        return input_tensor, notes_removed_count, notes_added_count
