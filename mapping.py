"""Mapping."""

import face_alignment.preprocess
import face_alignment.predictor
import face_alignment.model.baseline
import face_alignment.model.deep_ensemble
import face_alignment.model.s_MoE
import face_alignment.model.t_MoE

import facial_expression_recognition.preprocess
import facial_expression_recognition.predictor
import facial_expression_recognition.model.VGG16
import facial_expression_recognition.model.VGGFace
import facial_expression_recognition.model.deep_ensemble
import facial_expression_recognition.model.t_MoE
import facial_expression_recognition.model.thin


PREPROCESS = {
  "face_alignment": face_alignment.preprocess,
  "facial_expression_recognition": facial_expression_recognition.preprocess
}


MODEL = {
  "face_alignment": {
    "baseline": face_alignment.model.baseline,
    "deep_ensemble": face_alignment.model.deep_ensemble,
    "s_MoE": face_alignment.model.s_MoE,
    "t_MoE": face_alignment.model.t_MoE
  },
  "facial_expression_recognition": {
    "VGG16": facial_expression_recognition.model.VGG16,
    "VGGFace": facial_expression_recognition.model.VGGFace,
    "deep_ensemble": facial_expression_recognition.model.deep_ensemble,
    "t_MoE": facial_expression_recognition.model.t_MoE,
    "thin": facial_expression_recognition.model.thin
  }
}


PREDICTOR = {
  "face_alignment": face_alignment.predictor,
  "facial_expression_recognition": facial_expression_recognition.predictor
}