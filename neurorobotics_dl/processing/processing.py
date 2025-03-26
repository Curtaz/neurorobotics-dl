from mne.io import read_raw_gdf
from mne import events_from_annotations
from numpy import array
from ..utils import fix_mat
from scipy.io import loadmat
import  numpy as np
from scipy.signal import butter,lfilter
from pandas import DataFrame

OFFSET = 0x8000
    
def read_gdf(spath,verbosity='error',raw_events=False):
    raw = read_raw_gdf(spath,verbose=verbosity)

    eeg = raw.get_data().T
    
    events,names = events_from_annotations(raw,verbose=verbosity)
    names = {v:int(k) for k,v in names.items()}
    events_pos = events[:,0]
    events_typ = [names[e] for  e in events[:,2]]

    events = {'POS':array(events_pos),'TYP':array(events_typ)}
    if not raw_events:
        events = get_events(events)

    header = {'SampleRate':raw.info['sfreq'],
              'EVENT': events,
              'ChannelNames':np.array(raw.info['ch_names']),
            }
    
    return eeg,header

def read_mat(spath):
    data = loadmat(spath)
    eeg = data['s']
    h = fix_mat(data['h'])

    return eeg,h

def filter_data(eeg,fs,reference=None,filt_order=2,fc_hp=None,fc_lp=None,fc_bp=None):

    if reference is not None:
        if isinstance(reference,str) and reference.lower() == 'CAR': # Apply Common Average Referencing
            eeg = eeg - eeg.mean(axis=1)
        elif isinstance(reference,np.ndarray): # Apply Laplacian Referencing           
            eeg = np.matmul(eeg,reference)

    if fc_bp is not None: # Apply bandpass filter
        b, a = butter(filt_order, 2 * np.array(fc_bp) / fs, "band")
        eeg = lfilter(b, a, eeg,axis=0)

    if fc_hp is not None: # Apply highpass filter
        b, a = butter(filt_order, 2 * fc_hp / fs, "high")
        eeg = lfilter(b, a, eeg,axis=0)

    if fc_lp is not None: # Apply lowpass filter
        b, a = butter(filt_order, 2 * fc_lp / fs, "low")
        eeg = lfilter(b, a, eeg,axis=0)

    return eeg


def get_events(events, OFFSET=0x8000):
    events = DataFrame(events)
    true_events = events.loc[events['TYP']<OFFSET].copy() # Keep event openings only

    # Compute event ends based on the position of event closure
    true_events['END'] = np.nan

    for e in true_events['TYP'].unique():
        ev_idx = events['TYP']==(e + OFFSET)
        true_events.loc[true_events['TYP']==e,'END'] = np.where(events[ev_idx]['POS'],events[ev_idx]['POS'],0)

    true_events['END'] = true_events['END'].astype(int)
    true_events['DUR'] = true_events['END']-true_events['POS'] #Compute event duration

    # Keep only relevant columns
    true_events = true_events[['TYP','POS','DUR',]]

    return true_events.reset_index()

def get_events_mat(events):
    events = {k:events[k] for k in ['POS','TYP','DUR']}
    events = DataFrame(events)
    for col in events.columns: events[col] = events[col].astype(int)
    
    # trial_types = events['TYP'][(events['TYP'].isin([900,901,783]))].values
    # events = events[events['TYP']==Event.CONT_FEEDBACK]
    # events['TYP'] = trial_types

    # Keep only relevant columns
    events = events[['TYP','POS','DUR',]]

    return events.reset_index()