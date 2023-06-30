#!/usr/bin/python3

import rospy
from TTS.api import TTS
from tts.srv import TTSRequest

rospy.init_node('tts_node')


def handle_tts_request(req):

    model_name = TTS.list_models()[0]
    tts = TTS(model_name)
    tts.tts_to_file(text=req.text,speaker=tts.speakers[0], language=tts.languages[0],file_path="output.wav")
    return {'success',True}


rospy.Service('tts_service',TTSRequest,handle_tts_request)
        

    
        
        