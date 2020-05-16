.PHONY: build-all

# Make sure all of Vulkan SDK environment variables are set. (VULKAN_SDK, LD_LIBRARY_PATH, VK_LAYER_PATH)

CXXFLAGS=-Wall  -I${VULKAN_SDK}/include -g -std=c++17 
LDFLAGS=-L${VULKAN_SDK}/lib `pkg-config --libs glfw3` -lvulkan
CXX=g++

SRC_FILES=$(shell find src/ -name "*.cpp")
OBJ_FILES=$(patsubst %.cpp, %.o, $(SRC_FILES))
TARGET=vk_hello_triangle

GLSL_COMPILER=glslc
VERTEX_SHADER_FILES=$(shell find shaders/ -name "*.vert")
FRAG_SHADER_FILES=$(shell find shaders/ -name "*.frag")
VERTEX_SPV_FILES=$(patsubst %.vert, %.spv, $(VERTEX_SHADER_FILES))
FRAG_SPV_FILES=$(patsubst %.frag, %.spv, $(FRAG_SHADER_FILES))

build: $(OBJ_FILES)
	$(CXX) -o $(TARGET) $^ $(CXXFLAGS) $(LDFLAGS)

build-shaders:
	$(GLSL_COMPILER) $(VERTEX_SHADER_FILES) -o $(VERTEX_SPV_FILES)	
	$(GLSL_COMPILER) $(FRAG_SHADER_FILES) -o $(FRAG_SPV_FILES)

build-all: build-shaders build

rebuild: clean build-all

test: rebuild
	./$(TARGET)

clean:
	rm -rf $(OBJ_FILES) $(TARGET) $(VERTEX_SPV_FILES) $(FRAG_SPV_FILES)
