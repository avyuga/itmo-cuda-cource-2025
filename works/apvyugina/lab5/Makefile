LD_LIBRARY_PATH=/home/anastasia/TensorRT-10.14.1.48/targets/x86_64-linux-gnu/lib/:/usr/lib/x86_64-linux-gnu/

build:
	rm -rf build/*
	cmake . -B build/
	cmake --build build/

drop_engine:
	rm -rf models/*.engine

run_video:
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)
	rm output.log
	./build/DetectionModelTRT models/yolov10s_dyn.onnx video videos/v_J41d0cHAfSM_c008.mp4 > output.log

run_images:
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)
	rm output.log
	./build/DetectionModelTRT models/yolov10s_dyn.onnx images assets/ > output.log

.PHONY: build drop_engine run_images run_video