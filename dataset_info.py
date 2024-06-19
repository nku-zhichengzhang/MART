import numpy as np


label_info = {}

label_info['ve8'] = {}
label_info['ve8']['emotion'] = np.array(['Anger', 'Anticipation', 'Disgust', 'Fear',  'Joy', 'Sadness', 'Surprise', 'Trust'])
label_info['ve8']['sentiment'] = np.array([0, 1, 0, 0, 1, 0, 1, 1])

label_info['ek6'] = {}
label_info['ek6']['emotion'] = np.array(['anger', 'disgust', 'fear',  'joy', 'sadness', 'surprise'])
label_info['ek6']['sentiment'] = np.array([0, 0, 0, 1, 0, 1])

label_info['perr'] = {}
label_info['perr']['emotion'] = np.array(["Neutral", "Mild", "Intimate", "Tense", "Hostile"])
label_info['perr']['sentiment'] = np.array([2, 1, 1, 0, 0])