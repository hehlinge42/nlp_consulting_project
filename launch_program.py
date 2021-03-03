
import PySimpleGUI as sg
from json import (load as jsonload, dump as jsondump)
from os import path

from run_all import run_all

"""
    A simple "settings" implementation.  Load/Edit/Save settings for your programs
    Uses json file format which makes it trivial to integrate into a Python program.  If you can
    put your data into a dictionary, you can save it as a settings file.

    Note that it attempts to use a lookup dictionary to convert from the settings file to keys used in 
    your settings window.  Some element's "update" methods may not work correctly for some elements.

    Copyright 2020 PySimpleGUI.com
    Licensed under LGPL-3
"""

SETTINGS_FILE = path.join(path.dirname(__file__), r'settings_file.cfg')
DEFAULT_SETTINGS = {'wordclouds': False, 'tfidf': False , 'theme': sg.theme(), 'embedding_technique' : 'all'}
# "Map" from the settings dictionary keys to the window's element keys
SETTINGS_KEYS_TO_ELEMENT_KEYS = {'wordclouds': '-WC-', 'tfidf': '-TFIDF-' , 'theme': '-THEME-', 'embedding_technique' : '-EMBED-'}

##################### Load/Save Settings File #####################
def load_settings(settings_file, default_settings):
    try:
        with open(settings_file, 'r') as f:
            settings = jsonload(f)
    except Exception as e:
        sg.popup_quick_message(f'exception {e}', 'No settings file found... will create one for you', keep_on_top=True, background_color='red', text_color='white')
        settings = default_settings
        save_settings(settings_file, settings, None)
    return settings


def save_settings(settings_file, settings, values):
    if values:      # if there are stuff specified by another window, fill in those values
        for key in SETTINGS_KEYS_TO_ELEMENT_KEYS:  # update window with the values read from settings file
            try:
                settings[key] = values[SETTINGS_KEYS_TO_ELEMENT_KEYS[key]]
            except Exception as e:
                print(f'Problem updating settings from window values. Key = {key}')

    with open(settings_file, 'w') as f:
        jsondump(settings, f)

    sg.popup('Settings saved')

##################### Make a settings window #####################
def create_settings_window(settings):
    sg.theme(settings['theme'])

    def TextLabel(text, size): return sg.Text(text+':', justification='r', size=size)

    layout = [  [sg.Text('Settings', font='Any 15')],
                [TextLabel('Save WordClouds (Y/N)?', size=(25,1)), sg.Input(key='-WC-', default_text='N')],
                [TextLabel('Save TFIDF (Y/N)?', size=(25,1)), sg.Input(key='-TFIDF-')],
                [TextLabel("Embedding Technique ('lsi', 'word2vec', 'fasttext', 'all')", size=(50,1)), sg.Input(key='-EMBED-') ],
                [TextLabel('Theme', size=(15,1)), sg.Combo(sg.theme_list(), size=(20, 20), enable_events=True, key='-THEME-')],
                [sg.Button('Save'), sg.Button('Exit')]  ]

    window = sg.Window('Settings', layout, keep_on_top=True, finalize=True, size=(500, 200))

    for key in SETTINGS_KEYS_TO_ELEMENT_KEYS:   # update window with the values read from settings file
        try:
            window[SETTINGS_KEYS_TO_ELEMENT_KEYS[key]].update(value=settings[key])
        except Exception as e:
            print(f'Problem updating PySimpleGUI window from settings. Key = {key}')

    return window

##################### Main Program Window & Event Loop #####################
def create_main_window(settings):
    sg.theme(settings['theme'])

    layout = [[sg.T('Tripadvisor Run All Application')],
              [sg.T('This is amazing')],
              [sg.B('Run Application'), sg.B('Exit'), sg.B('Change Settings')]]

    return sg.Window('Main Application', layout, size=(500, 200))


def main():
    window, settings = None, load_settings(SETTINGS_FILE, DEFAULT_SETTINGS )

    while True:             # Event Loop
        if window is None:
            window = create_main_window(settings)

        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == 'Change Settings':
            event, values = create_settings_window(settings).read(close=True)
            if event == 'Save':
                window.close()
                window = None
                save_settings(SETTINGS_FILE, settings, values)
        if event == 'Run Application':
            print("Running")
            window, settings = None, load_settings(SETTINGS_FILE, DEFAULT_SETTINGS )
            run_all(settings)
            break
    window.close()

main()