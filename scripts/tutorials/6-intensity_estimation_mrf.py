"""

Examples to show how to estimate of the hyperparameters governing the GMM prior distributions.
This in the case where you want to train contrast-specific versions of SynthSeg.
Beware, if you do so, your model will not be able to segment any contrast at test time !
We do not provide example images and associated label maps, so do not try to run this directly !




If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


from SynthSeg.estimate_priors import build_intensity_stats

# ---------------------------------------------- simple multi-modal case -----------------------------------------------

# Here we have multi-modal images, where every image contains all channels.
# Channels are supposed to be sorted in the same order for all subjects.
image_dir = '/local_mount/space/ladyyy/data/users/ashwink/python-codebases/SynthSeg/scripts/tutorials/image_folder/multi_modal_mrf'

# same as before
save_label_dir = '/local_mount/space/ladyyy/data/users/ashwink/python-codebases/SynthSeg/scripts/tutorials/image_folder/multi_modal_mrf_labels'
estimation_labels = '/local_mount/space/ladyyy/data/users/ashwink/python-codebases/SynthSeg/Example_Data/numpy_arrays/generation_labels.npy'
estimation_classes = '/local_mount/space/ladyyy/data/users/ashwink/python-codebases/SynthSeg/Example_Data/numpy_arrays/generation_classes.npy'
result_dir = './outputs_tutorial_6/estimated_priors_multi_modal'

build_intensity_stats(list_image_dir=image_dir,
                      list_labels_dir=save_label_dir,
                      estimation_labels=estimation_labels,
                      estimation_classes=estimation_classes,
                      result_dir=result_dir,
                      max_channel=9,
                      rescale=False)