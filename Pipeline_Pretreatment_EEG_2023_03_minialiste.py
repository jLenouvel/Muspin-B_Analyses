# %%
%reset

# %%
%matplotlib qt5

# %% [markdown]
# # Ressources
# https://mne.tools/stable/auto_tutorials/time-freq/50_ssvep.html#sphx-glr-auto-tutorials-time-freq-50-ssvep-py 
# 
# ### Pour le rejet en général
# https://mne.tools/0.15/auto_tutorials/plot_artifacts_correction_rejection.html 
# 
# ### Rejet de bads channels : nous on ne rejette pas, on fait une interpolation pour garder le meme nombre de channels/participant pour les analyses cross groupe
# https://mne.tools/0.15/generated/mne.pick_types.html#mne.
# ici explication de l'interpolation : https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html#sphx-glr-auto-tutorials-preprocessing-15-handling-bad-channels-py 
#  
# ### Rejet de "bads" (que l'on peut définir nous même) pour les epochs :
# https://mne.tools/0.15/generated/mne.Epochs.html#mne.Epochs
# 
# ## Preprocessing : https://digyt.github.io/automated_EEG_cleaning_comparison/ , https://mne.tools/stable/auto_tutorials/preprocessing/index.html 
# " There are also different types of EEG noise. Some types, like noisy channels can be removed relatively effortless by visual inspection, fixed exclusion thresholds, exclusion methods like RANSAC (Bigdely-Shamlo et al., 2015; implemented: Python, MATLAB), or even be repaired with methods like Sensor Noise Suppression (SNS; De Cheveigné & Simon, 2008; Implemented: MATLAB).
# 
# Frequently reoccuring patterns like blinks, eye movements, or line noise can be removed using Independent Component Analysis (ICA; Jutten & Herault, 1991; Implemented: Python, MATLAB A, B), which, as of today, is the most commonly used tool for EEG preprocessing and the only one that has gained general acceptance (as a quasi-standard).
# 
# However, there are also less frequently occuring bursts of noise, which will be harder to repair with ICA and (if not removed) can impede the ICA's performance. These sporadic bursts artifacts are mostly created by some type of muscular activity and/or spontaneous movement of the subjects/electrodes. They are usually removed manually, a process that requires a large amount of time and often accounts for a significant proportion of the entire analysis procedure. Due to the large amount of time spent on this process, there is a relatively strong call throughout the EEG community to automate this process.
# 
# Numerous algorithms have been proposed for the automated cleaning of burst noise.
# 
# Among the currently most prominent algorithms are AutoReject (Jas et al., 2016; Implemented: Python), developed for the MNE-Python framework, and (riemannian) Artifact Subspace Reconstruction ((r)ASR; Kothe & Makeig, 2013; Blum et al., 2019; Implemtened: Python A, B, C, MATLAB), which is used as the standard EEG cleaning Addon in the EEGLAB toolbox." 
# https://digyt.github.io/automated_EEG_cleaning_comparison/ 
# 
# Impossible de tout faire à la main : 40 paticipant.e.s x 2 sessions x 9 bloc (7 blocs + train + estimation) == 720 inspections manuelles.
# Du coup, sur 10 sujets pris au hasard, 2 blocs pris au hasard (soit 20 inspections, bloc B1 à B7, parceque train et estimation on ne le regardera pas forcément ?) je vais faire :
#     - dans un 1er temps ICA automatique + bruit à la main 
#     - dans un 2nd temps r(ASR) et/ou AutoReject (car il y a des papiers qui l'utilisent, c'est "validé". Les autres méthodes ont l'air bien mais pas de papiers dessus donc + compliqué de justifier leurs performances)
# Si, lors de la réunion de Lundi, les résultats entre les deux/3 méthodes semblent identiques : on prend les algo qui clean automatiquement le bruit. Sinon on garde les ICA + rejet à la main (en faisant reject_by_annotation)...(mais vous m'aidez avec les 720 inspections manuelles :) ?)
# 
# Pour retirer le bruit electirque etc. de manière automatique, SSP ? 
# "SSP: Signal-space projection (SSP) . The most common use of SSP is to remove noise from MEG signals when the noise comes from environmental sources (sources outside the subject’s body and the MEG system, such as the electromagnetic fields from nearby electrical equipment) and when that noise is stationary (doesn’t change much over the duration of the recording). However, SSP can also be used to remove biological artifacts such as heartbeat (ECG) and eye movement (EOG) artifacts. Examples of each of these are given below." https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html#tut-artifact-ssp 
# 
# Template (cas ICA pour faire sur plusisuers sujets) : mne.preprocessing.corrmap ?

# %% [markdown]
# ## 1)   Libraries importation, functions definition, value attribution of constants

# %%
# Librairies exportation
from mne.io import read_raw_brainvision as bvread
from numpy.fft import fft, fftfreq
from scipy import signal
from mne.time_frequency.tfr import morlet
from mne.viz import plot_filter, plot_ideal_filter
import mne
import numpy as np
import matplotlib.pyplot as plt 
import os.path
import glob
import pandas as pd

# %% [markdown]
# ## 2)   Verifications : All files needed are here ?
# Files needed : .vhdr for each blok /session/participantµ

# %%
MISSING_FILES = []
START = 1
END = 2
INTERVAL = 1
ALL_SUB = list(range(START, END +1, INTERVAL))

for SUBJECT in ALL_SUB:
    #print (SUBJECT)
    pass

    SES_NUM = [1,2]
    for SESSION in SES_NUM :
        #print(SESSION)
        pass
        
        BEGIN_PATH_SAVE_FIGURE = '/home/lenouvju-admin/Bureau/BIDS_linux/results/Result_Pipeline4_Pretreatment_EEG/sub-{sub:02d}/ses-{ses:02d}/'.format(sub= SUBJECT,ses = SESSION)

        BLOCK_NUMBER = [1,2,3,4,5,6,7]
        for BLOCK in BLOCK_NUMBER:
            #print (BLOCK)
            pass
            
            # print (SUBJECT, SESSION, BLOCK)
            PATH = '/home/lenouvju-admin/Bureau/BIDS_linux/derivatives/Pipeline4_Pretreatment_EEG/sub-{sub:02d}/ses-{ses:02d}/B{blk}/P{sub:02d}{ses:02d}_B{blk}.vhdr'.format(sub= SUBJECT,ses = SESSION, blk = BLOCK)
            isFile = os.path.isfile(PATH)
            if isFile == False:
                MISSING_FILES.append(PATH)
                #print('File does not exist :' + PATH)

if len(MISSING_FILES) > 0 :
    print (str(len(MISSING_FILES)) + ' files are missing. \nPlease check that all .vhdr files exists and are in the good path before continue.')

# %%
START = 1
END = 2
INTERVAL = 1
ALL_SUB = list(range(START, END +1, INTERVAL))

for SUBJECT in ALL_SUB:
    #print (SUBJECT)
    pass

    SES_NUM = [1,2]
    for SESSION in SES_NUM :

        Dataframe_P_S_B = pd.DataFrame()
        BEGIN_PATH_SAVE_FIGURE = '/home/lenouvju-admin/Bureau/BIDS_linux/results/Result_Pipeline4_Pretreatment_EEG/sub-{sub:02d}/ses-{ses:02d}/'.format(sub= SUBJECT,ses = SESSION)

        BLOCK_NUMBER = [1,2,3,4,5,6,7]
        for BLOCK in BLOCK_NUMBER:
            #print (BLOCK)
            pass
            
            # print (SUBJECT, SESSION, BLOCK)
            PATH = '/home/lenouvju-admin/Bureau/BIDS_linux/derivatives/Pipeline4_Pretreatment_EEG/sub-{sub:02d}/ses-{ses:02d}/B{blk}/P{sub:02d}{ses:02d}_B{blk}.vhdr'.format(sub= SUBJECT,ses = SESSION, blk = BLOCK)
            Participant = 'P{sub:02d}{ses:02d}_B{blk}'.format(sub = SUBJECT, ses = SESSION, blk = BLOCK)
            # Data Loading
            raw = mne.io.read_raw_brainvision(PATH, preload = True) #, eog=('EOGV', 'EOGH', 'ECG'), misc='auto', scale=1.0, preload=True, verbose=None)
            #print(raw.info)

            Dict_P_S_B = {'P_S_B': Participant , 'Bad_channels_before_correction': [], 'Sources_ICA_rejected': []}# P_S_B for PArticipant_Sesssion_Blok. 
            # Final Dataframe wanted : P_S_B_E_E : Participant_Session_Blok_Essai_Epoch
            #If projectors (PCA) are already present in the raw fif file, it will be added to the info dictionary automatically. To remove existing projectors, you can do : evoked.add_proj([], remove_existing=True)  
            ssp_projectors = raw.info['projs']

            raw.del_proj()
            # Set montage
            montage = mne.channels.make_standard_montage('easycap-M1')# Hesitation between M1 and M10, à vérifier au CHU
            raw.set_montage(montage, on_missing = 'warn', verbose=False)
            #fig = raw.plot_sensors(show_names=True)# en rouge les bads channels

            RAW_BEFORE_BAD_CHANNELS_REMOOVED = raw.copy()
            #RAW_BEFORE_BAD_CHANNELS_REMOOVED.info['bads']# on l'enlève/les enlève après
            if RAW_BEFORE_BAD_CHANNELS_REMOOVED.info['bads'] == []:
                RAW_BEFORE_BAD_CHANNELS_REMOOVED.info['bads'] = ['Nothing']

            Dict_P_S_B['Bad_channels_before_correction'] = RAW_BEFORE_BAD_CHANNELS_REMOOVED.info['bads']

            # Si rien de bizare, on interpolate de suite les bad channels (on les enlève pas, à discuter), c'est exeptionnel, normalement on enlève rien avant la fin de artifacts detection.
            raw = raw.interpolate_bads(reset_bads=False)

            EEG_CHANNELS_WITH_BADS_CHANNELS = [value for value in RAW_BEFORE_BAD_CHANNELS_REMOOVED .ch_names if value != "EOGV" and value != "EOGH" and value != "ECG"]
            EEG_CHANNELS  = [value for value in raw.ch_names if value != "EOGV" and value != "EOGH" and value != "ECG"]
            EOG_CHANNELS = [value for value in raw.ch_names if value == "EOGV" or value == "EOGH"]
            ECG_CHANNELS  = [value for value in raw.ch_names if value == "ECG"]
            E0G_ECG_CHANNELS= EOG_CHANNELS + ECG_CHANNELS
            PICKS = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)

            #Power Spectral Density (PSD) to see power line noise
            fig1 = raw.plot_psd(fmax = 120, picks=EEG_CHANNELS,spatial_colors = True)#, n_fft = 2**13)  # What ? Pourquoi ça va jusqu'à 500 Hz ??? car Nyquist. JE n'avais pas autant de hight frequency noise avant, si ? On va s'arreter à 100Hz (max gamma)
            path_fig1 = BEGIN_PATH_SAVE_FIGURE + 'Block_' + str(BLOCK)+ '_PSD_RAW_EEG_CHANNELS_bads_channels_interpolated'
            plt.savefig(path_fig1, transparent = False)

            fig2 = raw.plot_psd(fmax = 120, picks=EEG_CHANNELS_WITH_BADS_CHANNELS,spatial_colors = True)#, n_fft = 2**13)  # What ? Pourquoi ça va jusqu'à 500 Hz ??? car Nyquist. JE n'avais pas autant de hight frequency noise avant, si ? On va s'arreter à 100Hz (max gamma)
            path_fig2 = BEGIN_PATH_SAVE_FIGURE + 'Block_' + str(BLOCK)+ '_PSD_RAW_EEG_CHANNELS_bads_channels_included'
            plt.savefig(path_fig2, transparent = False)

            raw_F = raw.copy()
            l_p= 150. # low_pass filter with supresses noise above 150 Hz
            h_p= 0.2


            '''mne.io.filter.filter_data(raw, sfreq = sfreq, l_freq=None, h_freq=l_p_150Hz, picks= EEG_CHANNELS, filter_length = 'auto',
                                        fir_design='firwin', phase='zero', verbose=True)'''

            raw_Filtered = raw_F.filter(l_freq=h_p, h_freq=l_p, fir_design='firwin', picks= EEG_CHANNELS, filter_length = 'auto',method = 'fir',phase='zero', verbose=True)

            notches = [50, 85]
            raw_Filtered = raw_Filtered.notch_filter(notches, phase='zero-double', fir_design='firwin')

            # Filtre basse freq
            filt_raw_pour_ICA = raw_Filtered.copy().filter(l_freq=1., h_freq=None) 

            # Fitting ICA
            # Epochs used for fitting ICA should not be baseline-corrected. 
            # Because cleaning the data via ICA may introduce DC offsets, we suggest to baseline correct your data after cleaning (and not before), 
            # should you require baseline correction.
            ica = mne.preprocessing.ICA(n_components=0.99, method='picard', fit_params=dict(extended=False),max_iter='auto',random_state=97)# TO DO pourquoi  fit_param ?
            ica.fit(filt_raw_pour_ICA )# et non sur raw pour mieux voir meme si ica fit sur filt_raw
            raw_Filtered.load_data()

            # Plot des sources ICA
            ica_fig = ica.plot_sources(raw_Filtered, show_scrollbars=False)

            # Initialiser la liste des sources sélectionnées
            selected_sources = []

            # Fonction pour enregistrer les sources sélectionnées
            def onclick(event):
                # Vérifier si l'utilisateur a cliqué sur une source
                if event.inaxes is not None:
                    # Obtenir l'indice de la source cliquée
                    source_idx = int(np.round(event.ydata))
                    # Ajouter la source sélectionnée à la liste
                    selected_sources.append(source_idx)
                    print(f"Source {source_idx} sélectionnée.")
                    # Marquer la source comme sélectionnée sur le plot
                    event.inaxes.axhline(y=source_idx, color='red')
                    fig4.canvas.draw()

            # Connecter la fonction onclick au plot des sources ICA
            cid = ica_fig.canvas.mpl_connect('button_press_event', onclick)

            # Obtenir la Figure associée au plot des sources ICA
            fig4 = ica_fig.get_figure()

            # Attendre que l'utilisateur ait terminé de sélectionner les sources
            plt.show()

            # Ajouter les lignes rouges sélectionnées dans la Figure
            for source_idx in selected_sources:
                fig4.axes[0].axhline(y=source_idx, color='red')

            # Enregistrer la Figure avec les sources sélectionnées
            path_ica_fig = BEGIN_PATH_SAVE_FIGURE + 'Block_' + str(BLOCK)+ '_ICA_SELECTION'
            fig4.savefig(path_ica_fig, transparent=False)

            selected_sources_unique = list(set(selected_sources))
            print(selected_sources_unique)
            Dict_P_S_B['Sources_ICA_rejected'] = selected_sources_unique

            # %%
            ica.exclude = selected_sources_unique

            # %%
            raw_after_ICA = raw_Filtered.copy()
            ica.apply(raw_after_ICA)

            #Brut
            #raw.plot(duration = 60, n_channels=67, remove_dc= False)# 64 eeg channels + 1 ECG + 2 EOGHv/H
            #raw_after_ICA.plot(duration = 60, n_channels=67, remove_dc= False)

            raw_Filtered.plot_psd(fmax = 150, picks= EEG_CHANNELS,spatial_colors = True, n_fft = 2**13, xscale = 'log') # What ? Pourquoi ça va jusqu'à 500 Hz ??? JE n'avais pas autant de hight frequency noise avant, si ? On va s'arreter à 100Hz (max gamma)
            path_raw_Filtered = BEGIN_PATH_SAVE_FIGURE + 'Block_' + str(BLOCK)+ '_PSD_LARGE_RAW_FILTERED_EEG_CHANNELS'
            plt.savefig(path_raw_Filtered, transparent = False)

            raw_after_ICA.plot_psd(fmax = 150, picks= EEG_CHANNELS,spatial_colors = True, n_fft = 2**13, xscale = 'log')
            path_raw_after_ICA = BEGIN_PATH_SAVE_FIGURE + 'Block_' + str(BLOCK)+ '_PSD_LARGE_RAW_AFTER_ICA_EEG_CHANNELS'
            plt.savefig(path_raw_after_ICA, transparent = False)

            #Power Spectral Density (PSD) to see power line noise
            raw_Filtered.plot_psd(fmax = 120, picks=EEG_CHANNELS,spatial_colors = True, n_fft = 2**13) # What ? Pourquoi ça va jusqu'à 500 Hz ??? JE n'avais pas autant de hight frequency noise avant, si ? On va s'arreter à 100Hz (max gamma)
            path_raw_after_Filtered = BEGIN_PATH_SAVE_FIGURE + 'Block_' + str(BLOCK)+ '_PSD_RAW_FILTERED_EEG_CHANNELS'
            plt.savefig(path_raw_after_Filtered, transparent = False)

            raw_after_ICA.plot_psd(fmax = 120, picks=EEG_CHANNELS,spatial_colors = True, n_fft = 2**13)
            path_raw_after_ICA = BEGIN_PATH_SAVE_FIGURE + 'Block_' + str(BLOCK)+ '_PSD_RAW_AFTER_ICA_EEG_CHANNELS'
            plt.savefig(path_raw_after_ICA, transparent = False)


            Dataframe_P_S_B = Dataframe_P_S_B.append(Dict_P_S_B, ignore_index=True)
        Dataframe_P_S_B.to_csv(BEGIN_PATH_SAVE_FIGURE) 


