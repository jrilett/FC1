# ----------------------------------------------------------------------------------------------------------------------

# Imports
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import gspread
import base64
from oauth2client.service_account import ServiceAccountCredentials
from pandas.io.json import json_normalize

# ----------------------------------------------------------------------------------------------------------------------

# Layout functions
def _max_width_():
    """
    Streamlit is fitted to the users screen resolution
    """
    max_width_str = f"max-width: 1000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )
_max_width_()

# Title
st.title('FILM CLUB STATISTICS (EST. 2020)')
st.text('')
st.text('')
st.markdown("""
A statistical exploration of Film Club, a growing record of over 600 films.
Numbers are pulled automatically from a google sheet.

James Rilett / Leo Loman / Tom Naccarato
""")

# ----------------------------------------------------------------------------------------------------------------------

# Google API and DF Build
scope = ['https://spreadsheets.google.com/feeds']
credentials = ServiceAccountCredentials.from_json_keyfile_name('./cool-eye-301417-5b2b82c69fbb.json', scope)
gc = gspread.authorize(credentials)
spreadsheet_key = '1_6mn3Dq77vlhhcgwoijQvuX0c-tghOPZUQT2PE-HxuM'
book = gc.open_by_key(spreadsheet_key)
worksheet = book.worksheet("Sheet1")
table = worksheet.get_all_values()
films = pd.DataFrame(table[1:], columns=table[0])
films = films[['Film', 'Year', 'Directors', 'Genre', 'Seen J', 'Review J', 'Seen L', 'Review L', 'Seen N', 'Review N']]
films = films.replace(r'^\s*$', np.nan, regex=True)

# ----------------------------------------------------------------------------------------------------------------------

# Formatting
films['Review J'] = films['Review J'].str.slice(0,2)
films['Review L'] = films['Review L'].str.slice(0,2)
films['Review N'] = films['Review N'].str.slice(0,2)
films['Seen J'].fillna(value = False, inplace = True)
films['Seen L'].fillna(value = False, inplace = True)
films['Seen N'].fillna(value = False, inplace = True)
films.replace({'Yes': True, 'YES': True}, inplace = True)

# ----------------------------------------------------------------------------------------------------------------------

# Scores DataFrames
def scoresdfbuilder(i, df = films):
    """ This function receives an initial and
    generates a scores dataframe which utilises
    the respective Review (initial) column. """
    ifilms = df[['Film', 'Year', 'Directors', 'Genre', f'Seen {i}', f'Review {i}']]
    ifilms = ifilms[ifilms[f'Review {i}'].notna()]
    ifilms[f'Review {i}'] = ifilms[f'Review {i}'].str.replace('BL', '11')
    ifilms[f'Review {i}'] = ifilms[f'Review {i}'].astype(str).astype(int)
    ifilms.Year = ifilms.Year.astype(str).astype(int)
    ifilms['Decade'] = (ifilms.Year//10)*10
    imean = ifilms[f'Review {i}'].mean()
    imean = round(imean,2)

    return ifilms, imean

jfilms, jmean = scoresdfbuilder('J')
lfilms, lmean = scoresdfbuilder('L')
nfilms, nmean = scoresdfbuilder('N')

# ----------------------------------------------------------------------------------------------------------------------

# Gif
st.text('')
st.text('')
file_ = open('filmgif1.gif', 'rb')
contents = file_.read()
data_url = base64.b64encode(contents).decode('utf-8')
st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------------------------------------------------

# Visualisation
## Mean Scores
st.markdown('***')
st.title('Mean Scores')
st.text('')
st.text('')

meansdata = {'Name': ['James', 'Leo', 'Naccers'], 'Mean Score': [jmean, lmean, nmean]}
meansdf = pd.DataFrame(meansdata)

st.write('James\' average film rating is {}'.format(meansdf.iloc[0,1]))
progress_bar = st.progress(0)
status_text = st.empty()
progress_bar.progress(int(meansdf.iloc[0,1])*10)

st.write('Leo\'s average film rating is {}'.format(meansdf.iloc[1,1]))
progress_bar = st.progress(0)
status_text = st.empty()
progress_bar.progress(int(meansdf.iloc[1,1])*10)

st.write('Naccers\' average film rating is {}'.format(meansdf.iloc[2,1]))
progress_bar = st.progress(0)
status_text = st.empty()
progress_bar.progress(int(meansdf.iloc[2,1])*10)

# ----------------------------------------------------------------------------------------------------------------------

## Scores Analysis

######################################################
#
#  BioSignalML Management in Python
#
#  Copyright (c) 2010  David Brooks
#
#  $Id: webstream.py,v a82ffb1e85be 2011/02/03 04:16:28 dave $
#
######################################################

import os
import uuid
import logging
import threading
import functools
import time

import tornado
from tornado.options import options
from tornado.websocket import WebSocketHandler
from tornado.ioloop import IOLoop

import biosignalml.transports.stream as stream

from biosignalml      import BSML
from biosignalml.rdf  import Uri
from biosignalml.data import TimeSeries, UniformTimeSeries
from biosignalml.data.convert import RateConverter
from biosignalml.units.convert import UnitConverter
import biosignalml.formats as formats
import biosignalml.utils as utils

from frontend import user


class StreamServer(WebSocketHandler):
#====================================

  protocol = 'biosignalml-ssf'

  def __init__(self, *args, **kwds):
  #---------------------------------
    WebSocketHandler.__init__(self, *args, **kwds)  ## Can't use super() as class is not
                                                    ## correctly initialised.
    self._parser = stream.BlockParser(self.got_block, check=stream.Checksum.CHECK)
    self._repo = options.repository
    self._capabilities = [ ]

  def select_subprotocol(self, protocols):
  #---------------------------------------
    if StreamServer.protocol in protocols:
      return StreamServer.protocol
    else:
      self.close()

  def got_block(self, block):
  #--------------------------
    pass

  def on_message(self, msg):
  #-------------------------
    self._capabilities = user.capabilities(self, None)
    try:
      bytes = bytearray(msg)
    except TypeError:
      bytes = bytearray(str(msg))
    #logging.debug('RAW: %s', bytes)
    self._parser.process(bytes)

  def send_block(self, block, check=stream.Checksum.STRICT):
  #---------------------------------------------------------
    '''
    Send a :class:`~biosignalml.transports.stream.StreamBlock` over a web socket.
    :param block: The block to send.
    :param check: Set to :attr:`~biosignalml.transports.stream.Checksum.STRICT`
      to append a SHA1 checksum to the block.
    '''
    if self.ws_connection is not None:
      self.write_message(str(block.bytes(check)), True)

  def close(self, *args):
  #----------------------
    if self.ws_connection is not None:
      WebSocketHandler.close(self, *args)

  def writing(self):
  #-----------------
    return self.ws_connection is not None and self.ws_connection.stream.writing()

  def write_count(self):
  #---------------------
    return len(self.ws_connection.stream._write_buffer) if self.ws_connection is not None else 0


class StreamEchoSocket(StreamServer):
#====================================

  def got_block(self, block):
  #--------------------------
    self.send_block(block)


class SignalReadThread(threading.Thread):
#========================================

  def __init__(self, handler, block, signals, rates, unit_map=None):
  #-----------------------------------------------------------------
    threading.Thread.__init__(self)
    self._handler = handler
    self._reqblock = block
    self._signals = signals
    self._rates = rates
    self._unit_map = unit_map

  def run(self):
  #-------------
    try:
      header = self._reqblock.header
      dtypes = { 'dtype': header.get('dtype'), 'ctype': header.get('ctype') }
      start = header.get('start')
      duration = header.get('duration')
      if start is None and duration is None: interval = None
      else:                                  interval = (start, duration)
      offset = header.get('offset')
      count = header.get('count')
      if offset is None and count is None: segment = None
      else:                                segment = (offset, count)
      maxpoints = header.get('maxsize', 0)

      # Interleave signal blocks...
      ### What if signal has multiple channels? What does read() return??
      sources = [ sig.read(interval=sig.recording.interval(*interval) if interval else None,
                   segment=segment, maxpoints=maxpoints) for sig in self._signals ]
      ## data is a list of generators
      starttimes = [ None ] * len(sources)
      converters = [ None ] * len(sources)
      datarate   = [ None ] * len(sources)
      self._active = len(sources)
      while self._active > 0:
        for n, sigdata in enumerate(sources):
          if sigdata is not None:
            try:
              data = sigdata.next()
              starttimes[n] = data.starttime
              siguri = str(self._signals[n].uri)
              keywords = dtypes.copy()
              if self._unit_map is not None: datablock = self._unit_map[n](data.data)
              else:                          datablock = data.data
              if data.is_uniform:
                keywords['rate'] = self._rates[n]
                if self._rates[n] != data.rate and converters[n] is None:
                  converters[n] = RateConverter(self._rates[n], data.data.size/len(data), maxpoints)
              else:
                if self._rates[n] is not None: raise ValueError("Cannot rate convert non-uniform signal")
                keywords['clock'] = data.times
              if converters[n] is not None:
                datarate[n] = data.rate
                for out in converters[n].convert(datablock, rate=data.rate):
                  self._send_block(stream.SignalData(siguri, starttimes[n], out, **keywords).streamblock())
                  starttimes[n] += len(out)/converters[n].rate
              else:
                self._send_block(stream.SignalData(siguri, starttimes[n], datablock, **keywords).streamblock())
            except StopIteration:
              if converters[n] is not None:
                for out in converters[n].convert(None, rate=datarate[n], finished=True):
                  self._send_block(stream.SignalData(siguri, starttimes[n], out, **keywords).streamblock())
                  starttimes[n] += len(out)/converters[n].rate
                converters[n] = None
              sources[n] = None
              self._active -= 1

    except Exception, msg:
      if str(msg) != "Stream is closed":
        logging.error("Stream exception - %s" % msg)
        self._send_block(stream.ErrorBlock(self._reqblock, str(msg)))
        if options.debug: raise

    finally:
      IOLoop.instance().add_callback(self._finished)

  def _send_block(self, block):
  #----------------------------
    IOLoop.instance().add_callback(functools.partial(self._send, block))

  def _send(self, block):
  #----------------------
    self._handler.send_block(block)

  def _finished(self):
  #-------------------
    if self._handler.write_count() > 0:
      stream = self._handler.ws_connection.stream ;
      stream._add_io_state(stream.io_loop.WRITE)
      IOLoop.instance().add_callback(self._finished)
      return
    self._active = -1
    self._handler.close()     ## All done with data request
    for s in self._signals: s.recording.close()


class StreamDataSocket(StreamServer):
#====================================

  MAXPOINTS = 50000

  def _send_error(self, msg):
  #--------------------------
    logging.error("Stream error: %s" % msg)
    self.send_block(stream.ErrorBlock(self._block, str(msg)))
    self.close()

  def _add_signal(self, uri):
  #--------------------------
    if self._repo.has_signal(uri):
      rec = self._repo.get_recording(uri, with_signals=False, open_dataset=False)
      recclass = formats.CLASSES.get(str(rec.format))
      if recclass:
        sig = self._repo.get_signal(uri,
           signal_class=rec.SignalClass)  ## Hack....!!
        rec.add_signal(sig)
        #print sig.graph.serialise()
        recclass.initialise_class(rec)
        self._sigs.append(sig)
      else:
        return self._send_error('No format for: %s' % uri)
    else:
      self._send_error('Unknown signal: %s' % uri)

  def _check_authorised(self, action):
  #-----------------------------------
    if action in self._capabilities: return True
    else:
      self._send_error("User <%s> not allowed to %s" % (self.user, user.ACTIONS[action]))
      return False

  @tornado.web.asynchronous
  def got_block(self, block):
  #--------------------------
    ##logging.debug('GOT: %s', block)
    self._block = block        ## For error handling
    if   block.type == stream.BlockType.ERROR:
      self.send_block(block)   ## Error blocks from parser v's from client...
    if   block.type == stream.BlockType.DATA_REQ:
      try:
        if not self._check_authorised(user.ACTION_VIEW): return
        uri = block.header.get('uri')
        ## Need to return 404 if unknown URI... (not a Recording or Signal)
        self._sigs = [ ]
        if isinstance(uri, list):
          for s in uri: self._add_signal(s)
        elif self._repo.has_recording(uri):
          rec = self._repo.get_recording(uri, with_signals=False)
          recclass = formats.CLASSES.get(str(rec.format))
          if recclass:
            recclass.initialise_class(rec)
            self._sigs = rec.signals()
        else:
          self._add_signal(uri)
        requested_rate = block.header.get('rate')
        if requested_rate is not None: rates = len(self._sigs)*[requested_rate]
        else:                          rates = [sig.rate for sig in self._sigs]

        unit_converter = UnitConverter(options.sparql_store)
        conversions = []
        requested_units = block.header.get('units')
        if requested_units is not None:
          units = len(self._sigs)*[requested_units]
          unit_map = [ unit_converter.mapping(sig.units, requested_units) for sig in self._sigs ]
        else:
          units = [str(sig.units) for sig in self._sigs],
          unit_map = None
#        self.send_block(stream.InfoBlock(channels = len(self._sigs),
#                                         signals = [str(sig.uri) for sig in self._sigs],
#                                         rates = rates,
#                                         units = units ))
        sender = SignalReadThread(self, block, self._sigs, rates, unit_map)
        sender.start()

      except Exception, msg:
        if str(msg) != "Stream is closed":
          self._send_error(msg)
          ##if options.debug: raise

#    elif block.type == stream.BlockType.INFO:
#      self._last_info = block.header
#      try:
#        uri = self._last_info['recording']
#        fname = self._last_info.get('dataset')
#
#        if fname is None:
#          fname = options.recordings_path + 'streamed/' + str(uuid.uuid1()) + '.h5'
#        ## '/streamed/' should come from configuration
#
#        ## Metaata PUT should have created file...
#
#        ## We support HDF5, user can use resource endpoint to PUT their EDF file...
#        recording = formats.hdf5.HDF5Recording(uri, dataset=fname)
#        # This will create, so must ensure that path is in our recording's area...
#
#        units = self._last_info.get('units')
#        rates = self._last_info.get('rates')
#        if self._last_info.get('signals'):
#          for n, s in enumerate(self._last_info['signals']):
#            recording.new_signal(siguri, units[n] if units else None,
#                                 rate=(rates[n] if rates else None) )
#        else:
#          signals = [ ]
#          for n in xrange(self._last_info['channels']):
#            signals.append(str(recording.new_signal(None, units[n] if units else None,
#                                 id=n, rate=(rates[n] if rates else None)).uri))
#          self._last_info['signals'] = signals
#        options.repository.store_recording(recording)
#        recording.close()
#
#      except Exception, msg:
#        self.send_block(stream.ErrorBlock(block, str(msg)))
#        if options.debug: raise

    elif block.type == stream.BlockType.DATA:
      # Got 'D' segment(s), uri is that of signal, that should have a recording link
      # look signal's uri up to get its Recording and hence format/source
      try:
        if not self._check_authorised(user.ACTION_EXTEND): return
        sd = block.signaldata()

#        if not sd.uri and sd.info is not None:
#          sd.uri = self._last_info['signals'][sd.info]
#        elif sd.uri not in self._last_info['signals']:
#          raise stream.StreamException("Signal '%s' not in Info header" % sd.uri)

        ## Extend to have mutiple signals in a block -- sd.uri etc are then lists

        ## Also get and use graph uri...
        rec_graph, rec_uri = self._repo.get_graph_and_recording_uri(sd.uri)
        if rec_uri is None or not self._repo.has_signal(sd.uri, rec_graph):
          raise stream.StreamException("Unknown signal '%s'" % sd.uri)

        rec = self._repo.get_recording(rec_uri, open_dataset=False, graph_uri=rec_graph)
        if str(rec.format) != formats.MIMETYPES.HDF5:
          raise stream.StreamException("Signal can not be appended to -- not HDF5")

        if rec.dataset is None:
          rec.dataset = os.path.join(options.recordings_path, str(uuid.uuid1()) + '.h5')
          self._repo.insert_triples(rec_graph,
            [ ('<%s>' % rec_uri, '<%s>' % BSML.dataset, '<%s>' % utils.file_uri(rec.dataset)) ])
          rec.initialise(create=True)
        else:
          rec.initialise()  # Open hdf5 file

        if sd.rate: ts = UniformTimeSeries(sd.data, rate=sd.rate)
        else:       ts = TimeSeries(sd.data, sd.clock)

        sig = rec.get_signal(sd.uri)
        sig.initialise(create=True, dtype=sd.dtype)
        # what if sd.units != sig.units ??
        # what if sd.rate != sig.rate ??
        # What if sd.clock ??

        sig.append(ts)

        if rec.duration is None or rec.duration < sig.duration:
          rec.duration = sig.duration
          self._repo.save_subject_property(rec_graph, rec, 'duration')
        rec.close()

      except Exception, msg:
        self.send_block(stream.ErrorBlock(block, str(msg)))
        if options.debug: raise


if __name__ == '__main__':
#=========================

  import sys

  import biosignalml.repository as repository

  def print_object(obj):
  #=====================
    attrs = [ '', repr(obj) ]
    for k in sorted(obj.__dict__):
      attrs.append('  %s: %s' % (k, obj.__dict__[k]))
    print '\n'.join(attrs)


  def test(uri):
  #-------------
    repo = repository.BSMLRepository('http://devel.biosignalml.org', 'http://localhost:8083')

  if len(sys.argv) < 2:
    print "Usage: %s uri..." % sys.argv[0]
    sys.exit(1)

  uri = sys.argv[1:]
  if len(uri) == 1: uri = uri[0]

  test(uri)