# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: speech.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0cspeech.proto\x12\x06speech\"|\n\rSpeechRequest\x12\x12\n\naudio_data\x18\x01 \x01(\x0c\x12\x10\n\x08language\x18\x02 \x01(\t\x12\x13\n\x0bsample_rate\x18\x03 \x01(\x05\x12\x1c\n\x0finterim_results\x18\x04 \x01(\x08H\x00\x88\x01\x01\x42\x12\n\x10_interim_results\"Z\n\x0eSpeechResponse\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x10\n\x08is_final\x18\x02 \x01(\x08\x12\x12\n\nconfidence\x18\x03 \x01(\x02\x12\x14\n\x0c\x61lternatives\x18\x04 \x03(\t2\x91\x01\n\x06Speech\x12<\n\tRecognize\x12\x15.speech.SpeechRequest\x1a\x16.speech.SpeechResponse\"\x00\x12I\n\x12StreamingRecognize\x12\x15.speech.SpeechRequest\x1a\x16.speech.SpeechResponse\"\x00(\x01\x30\x01\x42!\n\x12\x63om.tyg.speech.rpcB\x0bSpeechProtob\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'speech_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\022com.tyg.speech.rpcB\013SpeechProto'
  _globals['_SPEECHREQUEST']._serialized_start=24
  _globals['_SPEECHREQUEST']._serialized_end=148
  _globals['_SPEECHRESPONSE']._serialized_start=150
  _globals['_SPEECHRESPONSE']._serialized_end=240
  _globals['_SPEECH']._serialized_start=243
  _globals['_SPEECH']._serialized_end=388
# @@protoc_insertion_point(module_scope)
