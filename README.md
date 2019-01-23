# fenghuo
FengHuo Gesture Recognition System.

# dir tree

```bash
.
├── LICENSE
├── README.md
├── fenghuo-client #client code
│   ├── callbacks
│   │   └── callback.py #event callbacks
│   ├── context
│   │   └── context.py
│   ├── hand_inference_graph
│   │   ├── frozen_inference_graph.pb
│   │   └── hand_label_map.pbtxt.txt
│   ├── protocol
│   │   └── predict_protocol.py
│   ├── requirements.txt
│   ├── simhei.ttf
│   ├── start.py
│   ├── start_v2.py
│   ├── start_v2.py.bak
│   └── utils
│       ├── detector_utils.py
│       └── label_map_util.py
└── fenghuo-server #server code
    ├── c3d_model_deep_deep.py
    ├── configs
    │   └── model_config.json
    ├── model
    │   └── model_best.pth.tar #pre-trained gesture recognition model.
    ├── protocol # data exchange protocol between client and server.
    │   └── predict_protocol.py
    ├── readme.md
    ├── remote_server.py
    ├── remote_server.py.bak
    └── requirements.txt

```
