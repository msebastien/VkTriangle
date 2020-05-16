#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

/* Platform-specific surface stuff
#if defined(__linux__)
    #include <vulkan/vulkan_xcb.h> // X11 window
#elif defined(_WIN32)
    #include <Windows.h>
    #include <vulkan/vulkan_win32.h>
#endif
*/

#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <set>
#include <cstdint> // UINT32_MAX
#include <algorithm>
#include <fstream>

#define WIDTH 800
#define HEIGHT 600

// Instance level
std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// device-specific extensions
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME // Swap chain which will contain framebuffers
};

const int MAX_FRAMES_IN_FLIGHT = 2; // defines how many frames should be processed concurrently

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

// Debug Utils
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, 
const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if(func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if(func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}


class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;
    VkInstance instance;

    VkSurfaceKHR surface;  // surface to present rendered images

    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
     
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;

    // Swap Chain related stuff
    VkSwapchainKHR swapChain;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    // Pipeline
    std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
    };
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout; // "uniform" variables for shaders (dynamic values = no need to recreate the shaders)
    VkPipeline graphicsPipeline;

    // Command buffers
    VkCommandPool commandPool; // command pool for the graphics queue family which allocates command buffers
    std::vector<VkCommandBuffer> commandBuffers;
    
    // Synchronize queue operations of draw commands and presentation
    // Each frame should have its own set of semaphores (and fences) :
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    // To perform CPU-GPU synchronization, Vulkan offers a second type of synchronization primitive called fences. 
    // Fences are similar to semaphores in the sense that they can be signaled and waited for, but this time we actually wait for them in our own code.
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;
    size_t currentFrame = 0;
    bool frameBufferResized = false;
    
    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, frameBufferResizeCallback);
    }

    static void frameBufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->frameBufferResized = true;
    }

    //----------------------------------------------------------------------------------------------------------
    //  VULKAN INSTANCE
    //----------------------------------------------------------------------------------------------------------
    
    void createInstance() {
        if(enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        // App information for Vulkan driver (optional)
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle (Vulkan)";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;   
        
        // Retrieve Vulkan global extensions to use
        auto extensions = getRequiredExtensions();
        //checkExtensionSupport();
        
        // Set global extensions to use
        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();
        
        // Set validation layers to use and debug messenger specifics
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
        if(enableValidationLayers){ // If debugging/validation layers are enabled
            // layers
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            // debug messenger for debugging vkCreateInstance and vkDestroyInstance
            // pNext => nullptr or pointer to an Extension-specific structure
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;

        } else { // If debugging is disabled
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }
           
        // Create a Vulkan instance
        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
        if(result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create a vulkan instance!");
        }
    }

    void checkExtensionSupport() {
        // Get the number of extensions
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

        // Get Extension properties
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        std::cout << "available extensions:" << std::endl;
        for(const auto& extension : extensions) {
            std::cout << "\t" << "Name: " << extension.extensionName << std::endl;
            std::cout << "\t\t" << "SpecVersion: " << extension.specVersion << std::endl;
        }
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if(enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    bool checkValidationLayerSupport() {
        // Get the number of available layers
        uint32_t layerCount = false;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        // Retrieve available layer properties
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        // Check if the requested layers are available 
        for(const char* layerName : validationLayers) {
            bool layerFound = false;

            for(const auto& layerProperties : availableLayers) {
                if(strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }
            
            // if the layer has not been found, stops iterating list and return false
            if(!layerFound){
                return false;
            }
        }

        return true;
    }
    //-----------------------------------------------------------------------------------------------

    //-----------------------------------------------------------------------------------------------
    //  DEBUGGING FUNCTIONS
    //-----------------------------------------------------------------------------------------------

    // Debug callback function
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData) {

        std::cerr << "Validation Layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE; // everything is ok, else return VK_TRUE
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo){
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

        createInfo.pfnUserCallback = debugCallback;
        createInfo.pUserData = nullptr; // optional 
    }

    void setupDebugMessenger() {
        if(!enableValidationLayers) return; // if validation layers disabled, do nothing

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if(CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }
    //----------------------------------------------------------------------------------------------

    //----------------------------------------------------------------------------------------------
    //  WINDOW SURFACE
    //----------------------------------------------------------------------------------------------

    /* Already implemented in GLFW (x11_window.c line 3096)
    void createSurface() {
    #if defined(__linux__)
        VkXcbSurfaceCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
        createInfo.connection = 

        if (vkCreateXcbSurfaceKHR(instance, &createInfo, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface!");
        }

    #elif defined(_WIN32)
        VkWin32SurfaceCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        createInfo.hwnd = glfwGetWin32Window(window);
        createInfo.hinstance = GetModuleHandle(nullptr);

        if (vkCreateWin32SurfaceKHR(instance, &createInfo, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface!");
        }
    #endif
    }*/

    void createSurface() {
        if(glfwCreateWindowSurface(instance, window, nullptr, &surface)) {
            throw std::runtime_error("Failed to create window surface!");
        }
    }

    //----------------------------------------------------------------------------------------------
    //  PHYSICAL DEVICES & QUEUE FAMILIES
    //----------------------------------------------------------------------------------------------
    void pickPhysicalDevice() {
        // Get the number of devices
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if(deviceCount == 0) {
            throw std::runtime_error("Failed to find GPUs with Vulkan support!");
        }

        // Retrieve a list of physical devices
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for(const auto& device : devices) {
            if(isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }
    }

    /*bool isDeviceSuitable(VkPhysicalDevice device) {
        VkPhysicalDeviceProperties deviceProperties;
        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && 
        deviceFeatures.geometryShader;
    }*/

    bool isDeviceSuitable(VkPhysicalDevice device) {
        // Queue families
        QueueFamilyIndices indices = findQueueFamilies(device);

        // Extensions
        bool extensionsSupported = checkPhysicalDeviceExtensionSupport(device);

        // Swapchain support
        bool swapChainAdequate = false;
        if(extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupportDetails(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    /* Other possible implementation
    bool checkPhysicalDeviceExtensionSupport(VkPhysicalDevice device) {
        // Retrieve the number of device-specific supported extensions
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        // Retrieve available device-specific extensions
        std::vector<VkExtensionProperties> availableDeviceExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableDeviceExtensions.data());

        // Compare with the the list of device extension that we want to load
        for(const char* deviceExtensionName : deviceExtensions) {
            bool extensionFound = false;

            for(const auto& deviceExtension : availableDeviceExtensions) {
                if(strcmp(deviceExtensionName, deviceExtension.extensionName)) {
                    extensionFound = true;
                    break;
                }
            }

            // If at least one extension is missing, quit function and return false
            if(!extensionFound) {
                return false;
            }

            return true;
        }
    }*/

    bool checkPhysicalDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    // Queue families to be used
    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete() {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        // Get the number of queue families which has the physical device
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        // Retrieve queue families for the current physical device
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for(const auto& queueFamily : queueFamilies) {
            // Retrieve index for the first queue family which supports "VK_QUEUE_GRAPHICS_BIT"
            if(queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            // Retrieve index for the first queue family which can present images
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if(presentSupport) {
                indices.presentFamily = i;
            }

            if(indices.isComplete()){
                break;
            }

            i++;
        }

        return indices;
    }
    //----------------------------------------------------------------------------------------------

    //----------------------------------------------------------------------------------------------
    //  LOGICAL DEVICE & QUEUES
    //----------------------------------------------------------------------------------------------
    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {
            indices.graphicsFamily.value(),
            indices.presentFamily.value()};
        
        // Queues creation infos
        float queuePriority = 1.0f;
        for(uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority; // required

            queueCreateInfos.push_back(queueCreateInfo);
        }
        
        // Logical device creation info
        VkPhysicalDeviceFeatures deviceFeatures = {};
        VkDeviceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;

        // Device-specific extensions (and validation layers => instance-level only)
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if(enableValidationLayers) { // deprecated (instance-level only)
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        // Create the logical device
        // The queues are automatically created along with the logical device,
        // but we now need a handle to interface with them
        // => Let's a create a new VkQueue class member to store a handle to the graphics queue.
        if(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create logical device!");
        }

        // Retrieve queue handle
        // We can use the vkGetDeviceQueue function to retrieve queue handles for each queue family. 
        // The parameters are the logical device, queue family, queue index and a pointer to the variable to store the queue handle in. 
        // Because we're only creating a single queue from this family, we'll simply use index 0.
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }
    //----------------------------------------------------------------------------------------------
    
    //----------------------------------------------------------------------------------------------
    //  SWAPCHAIN
    //----------------------------------------------------------------------------------------------
    
    struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capabilities; // min & max number of images, min & max width/height..
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    SwapChainSupportDetails querySwapChainSupportDetails(VkPhysicalDevice device) {
        SwapChainSupportDetails details; // struct

        // Capabilities
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
        
        // Formats
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if(formatCount > 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }
        
        // Present modes
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if(presentModeCount > 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }
    
    //-------------------------------------------//
    //  a. CHOOSE FORMAT & PRESENT MODE & EXTENT //
    //-------------------------------------------//
    // Each VkSurfaceFormatKHR entry contains a format and a colorSpace member. 
    // The format member specifies the color channels and types. For example, 
    // VK_FORMAT_B8G8R8A8_SRGB means that we store the B, G, R and alpha channels 
    // in that order with an 8 bit unsigned integer for a total of 32 bits per pixel. 
    // The colorSpace member indicates if the SRGB color space is supported or not 
    // using the VK_COLOR_SPACE_SRGB_NONLINEAR_KHR flag. For the color space, 
    // we'll use SRGB if it is available, because it results in more accurate perceived colors. 
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for(const auto& availableFormat : availableFormats) {
            if(availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }

    // The presentation mode is arguably the most important setting for the swap chain, 
    // because it represents the actual conditions for showing images to the screen.
    // 4 modes : VK_PRESENT_MODE_IMMEDIATE_KHR (causes tearing), VK_PRESENT_MODE_FIFO_KHR (vsync/vblank);
    // VK_PRESENT_MODE_FIFO_KHR has 2 variants : VK_PRESENT_MODE_FIFO_RELAXED_KHR (don't wait on vblank to transfer image, can cause tearing)
    // and VK_PRESENT_MODE_MAILBOX_KHR (instead of blocking the app, when queue is full, 
    // images are replaced with newer ones. Used to implement TRIPLE BUFFERING 
    // => avoid tearing with less latency issues than standard vsync) 
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for(const auto& availablePresentMode : availablePresentModes) {
            // check if the best mode is available
            if(availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR; // guaranteed to be available
    }

    // The swap extent is the resolution of the swap chain images and it's almost always
    // exactly equal to the resolution of the window that we're drawing to. The range of 
    // the possible resolutions is defined in the VkSurfaceCapabilitiesKHR structure. 
    // Vulkan tells us to match the resolution of the window by setting the width and 
    // height in the currentExtent member. However, some window managers do allow us to 
    // differ here and this is indicated by setting the width and height in currentExtent 
    // to a special value: the maximum value of uint32_t. In that case we'll pick the 
    // resolution that best matches the window within the minImageExtent and maxImageExtent bounds.
    // The max and min functions are used here to clamp the value of WIDTH and HEIGHT 
    // between the allowed minimum and maximum extents that are supported by the implementation. 
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if(capabilities.currentExtent.width != UINT32_MAX) {
            return capabilities.currentExtent;
        } else {
            // To handle window resizes properly, we also need to query the current size of the framebuffer 
            // to make sure that the swap chain images have the (new) right size.
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {static_cast<uint32_t>(width),
                                        static_cast<uint32_t>(height)}; // Actual Window resolution

            // Resolution that best matches the window within minImageExtent and maxImageExtent
            actualExtent.width = std::max( capabilities.minImageExtent.width,
            std::min(capabilities.maxImageExtent.width, actualExtent.width) );

            actualExtent.height = std::max( capabilities.minImageExtent.height, 
            std::min(capabilities.maxImageExtent.height, actualExtent.height) );

            return actualExtent;
        }
    }
    //-------------------------------------------// end of "choose format/present mode/swap extent"
    
    //----------------------------------//
    //  b. CREATING THE SWAP CHAIN      //
    //----------------------------------//

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupportDetails(physicalDevice);

        // Retrieve swap chain details
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        // save some swap chain properties in class member variables
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;

        // Simply sticking to this minimum means that we may sometimes have to wait on 
        // the driver to complete internal operations before we can acquire another image 
        // to render to. Therefore it is recommended to request at least one more image than the minimum.
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

        // We should also make sure to not exceed the maximum number of images while doing this, 
        // where 0 is a special value that means that there is no maximum
        if(swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }
        //-------

        VkSwapchainCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1; // always 1 unless this is a stereoscopic 3D app
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // specifies what kind of operations we'll use the images in the swap chain for.
        // It is also possible to render images to a separate image first to 
        // perform operations like post-processing. In that case to use a value 
        // like VK_IMAGE_USAGE_TRANSFER_DST_BIT instead and use a memory operation to 
        // transfer the rendered image to a swap chain image.

        
        // How to handle swap chain images that will be used accross multiple
        // queue families. That will be the case in our application if the graphics queue
        // family is different from the presentation queue. We'll be drawing on the images 
        // in the swap chain from the graphics queue and then submitting them on the presentation queue.
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        if(indices.graphicsFamily != indices.presentFamily) {
            // Images can be used across multiple queue families without explicit ownership transfer
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT; 
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            // An image is owned by one queue family at a time and ownership must be explicitly 
            // transfered before using it in another queue family. This option offers the best performance.
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0; // optional
            createInfo.pQueueFamilyIndices = nullptr; // optional
        }
        
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE; // If "clipped" is set to VK_TRUE then that means that we don't care about the color of pixels that are obscured.
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create the Swap Chain!");
        }

        // RETRIEVING SWAP CHAIN IMAGES

        // Get image count
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);

        // retrieve images
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
    }
    //----------------------------------------------------------------------------------------------

    //----------------------------------------------------------------------------------------------
    //  IMAGE VIEWS
    //----------------------------------------------------------------------------------------------
    
    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for(size_t i = 0; i < swapChainImages.size(); i++){
            VkImageViewCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];

            // The viewType and format fields specify how the image data should be interpreted. 
            // The viewType parameter allows to treat images as 1D textures, 2D textures, 3D textures and cube maps.
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;

            // The components field allows to swizzle the color channels around. 
            // For example, you can map all of the channels to the red channel for a monochrome texture.
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            // The subresourceRange field describes what the image's purpose is 
            // and which part of the image should be accessed. Our images will be used as 
            // color targets without any mipmapping levels or multiple layers.
            // If you were working on a stereographic 3D application, then you would create 
            // a swap chain with multiple layers. You could then create multiple image views 
            // for each image representing the views for the left and right eyes by accessing different layers.
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if(vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create image views!");
            }
        }
    }
    //---------------------------------------------------------------------------------------------- 
    
    // An image view is sufficient to start using an image as a texture, but it's not quite ready 
    // to be used as a render target just yet. That requires one more step of indirection, known as a framebuffer. 
    // But first we'll have to set up the graphics pipeline.

    //------------------------------------------------------------------------------------------
    // GRAPHICS PIPELINE
    //------------------------------------------------------------------------------------------
    // The graphics pipeline is the sequence of operations that take the vertices and textures 
    // of your meshes all the way to the pixels in the render targets.
    
    void createRenderPass() {
        VkAttachmentDescription colorAttachment = {};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; // for multisampling
        
        // The loadOp and storeOp determine what to do with the data in the attachment before rendering and after rendering.
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // clear values to a constant at the start
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // rendered contents will be stored in memory and can be read later
        
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; // existing contents are undefined, we don't ware about them
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        // Textures and framebuffers in Vulkan are represented by VkImage objects with a certain pixel format, 
        // however the layout of the pixels in memory can change based on what you're trying to do with an image.

        // Some of the most common layouts are:
        // ==> VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: Images used as color attachment
        // ==> VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: Images to be presented in the swap chain
        // ==> VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: Images to be used as destination for a memory copy operation
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        // Attachment reference
        VkAttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        // Subpass
        // A single render pass can consist of multiple subpasses. 
        // Subpasses are subsequent rendering operations that depend on the contents of framebuffers in previous passes, 
        // for example a sequence of post-processing effects that are applied one after another. 
        // If we group these rendering operations into one render pass, then Vulkan is able to reorder the operations 
        // and conserve memory bandwidth for possibly better performance. 
        // For our very first triangle, however, we'll stick to a single subpass.
        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        // The index of the attachment in this array is directly referenced from the fragment shader with the layout(location = 0) out vec4 outColor directive!
        // The following other types of attachments can be referenced by a subpass:
        // ==> pInputAttachments: Attachments that are read from a shader
        // ==> pResolveAttachments: Attachments used for multisampling color attachments
        // ==> pDepthStencilAttachment: Attachment for depth and stencil data
        // ==> pPreserveAttachments: Attachments that are not used by this subpass, but for which the data must be preserved

        // SubPass dependencies
        // Subpasses in a render pass automatically take care of image layout transitions. 
        // These transitions are controlled by subpass dependencies, which specify memory and execution dependencies between subpasses. 
        // We have only a single subpass right now, but the operations right before and right after this subpass also count as implicit "subpasses".
        // There are two built-in dependencies that take care of the transition at the start of the render pass and at the end of the render pass, 
        // but the former does not occur at the right time. It assumes that the transition occurs at the start of the pipeline, but we haven't acquired the image yet at that point! 
        // There are two ways to deal with this problem. We could change the waitStages for the imageAvailableSemaphore to VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT to ensure that 
        // the render passes don't begin until the image is available, or we can make the render pass wait for the VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT stage. 
        // I've decided to go with the second option here, because it's a good excuse to have a look at subpass dependencies and how they work.
        
        // The first two fields specify the indices of the dependency and the dependent subpass. The special value VK_SUBPASS_EXTERNAL refers to the implicit subpass before or after 
        // the render pass depending on whether it is specified in srcSubpass or dstSubpass. The index 0 refers to our subpass, which is the first and only one. 
        // The dstSubpass must always be higher than srcSubpass to prevent cycles in the dependency graph.
        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;

        // Specify the operations to wait on and the stages in which these operations occur. 
        // We need to wait for the swap chain to finish reading from the image before we can access it. 
        // This can be accomplished by waiting on the color attachment output stage itself.
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;

        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        // Render Pass
        // The render pass object can then be created by filling in the VkRenderPassCreateInfo structure with an array of attachments and subpasses. 
        // The VkAttachmentReference objects reference attachments using the indices of this array.
        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;
        
        if(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create render pass!");
        }
    }

    static std::vector<char> readFile(const std::string& fileName) {
        std::ifstream file(fileName, std::ios::ate | std::ios::binary);

        if(!file.is_open()) {
            throw std::runtime_error("Failed to open file!");
        }

        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);
    
        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();
    
        return buffer;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create shader module!");
        }

        return shaderModule;
    }

    void createGraphicsPipeline() {
        auto vertShaderCode = readFile("shaders/vertex-shader.spv");
        auto fragShaderCode = readFile("shaders/fragment-shader.spv");

        // SHADER MODULE
        // The compilation and linking of the SPIR-V bytecode to machine code for 
        // execution by the GPU doesn't happen until the graphics pipeline is created. 
        // That means that we're allowed to destroy the shader modules again as soon as 
        // pipeline creation is finished, which is why we'll make them local variables
        // instead of class members
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        // SHADER STAGES (programmable)
        // 1. for vertex shader
        VkPipelineShaderStageCreateInfo vertShaderStageCreateInfo = {};
        vertShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageCreateInfo.module = vertShaderModule;
        vertShaderStageCreateInfo.pName = "main"; // entrypoint function
        // .pSpecilizationInfo => allows to specify values for shader constants. 
        // You can use a single shader module where its behavior can be configured at 
        // pipeline creation by specifying different values for the constants used in it.

        // 2. for fragment shader
        VkPipelineShaderStageCreateInfo fragShaderStageCreateInfo = {}; 
        fragShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageCreateInfo.module = fragShaderModule;
        fragShaderStageCreateInfo.pName = "main"; // entrypoint function

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageCreateInfo, fragShaderStageCreateInfo};

        // FIXED-FUNCTION STATES (!= shaders)
        // 1. Vertex Input
        // The VkPipelineVertexInputStateCreateInfo structure describes the format of the vertex data that will be passed to the vertex shader. 
        // It describes this in roughly two ways:
        // ==> Bindings: spacing between data and whether the data is per-vertex or per-instance (see instancing)
        // ==> Attribute descriptions: type of the attributes passed to the vertex shader, which binding to load them from and at which offset
        VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr; // optional
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr; // optional

        // 2. Input Assembly (primitives)
        // The VkPipelineInputAssemblyStateCreateInfo struct describes two things: 
        // what kind of geometry will be drawn from the vertices and if primitive restart should be enabled.
        VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // 3. Viewports, scissors & Viewport State
        // Viewport basically describes the region of the framebuffer that the output will be rendered to. 
        // This will almost always be (0, 0) to (width, height)
        // the size of the swap chain and its images may differ from the WIDTH and HEIGHT of the window. 
        // The swap chain images will be used as framebuffers later on, so we should stick to their size.
        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) swapChainExtent.width;
        viewport.height = (float) swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        // While viewports define the transformation from the image to the framebuffer, scissor rectangles define in which regions pixels will actually be stored. 
        // Any pixels outside the scissor rectangles will be discarded by the rasterizer. They function like a filter rather than a transformation. 
        VkRect2D scissor = {};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;

        // Viewport State
        VkPipelineViewportStateCreateInfo viewportState = {};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        // 4. Rasterizer State
        // The rasterizer takes the geometry that is shaped by the vertices from the vertex shader and turns it into fragments to be colored by the fragment shader. 
        // It also performs depth testing (Z-Buffering), face culling and the scissor test, and it can be configured to output fragments that fill entire polygons or just the edges (wireframe rendering).
        VkPipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f; // thickness in number of fragments
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE; // specifies the vertex order for faces to be considered front-facing and can be clockwise or counterclockwise.
        
        // The rasterizer can alter the depth values by adding a constant value or biasing them based on a fragment's slope. 
        // This is sometimes used for shadow mapping
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f; // optional
        rasterizer.depthBiasClamp = 0.0f; // optional
        rasterizer.depthBiasSlopeFactor = 0.0f; // optional

        // 5. Multisampling State (Anti-aliasing)
        VkPipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f; // optional
        multisampling.pSampleMask = nullptr; // optional
        multisampling.alphaToCoverageEnable = VK_FALSE; // optional
        multisampling.alphaToOneEnable = VK_FALSE; // optional

        // 6. Depth & Stencil testing State
        // If you are using a depth and/or stencil buffer, then you also need to configure the depth and stencil tests using VkPipelineDepthStencilStateCreateInfo. 
        // We don't have one right now, so we can simply pass a nullptr instead of a pointer to such a struct. 
        // We'll get back to it in the depth buffering chapter.

        // 7. Color Blending State
        // After a fragment shader has returned a color, it needs to be combined with the color that is already in the framebuffer. 
        // This transformation is known as color blending and there are two ways to do it:
        // ==> Mix the old and new value to produce a final color
        // ==> Combine the old and new value using a bitwise operation
        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA; // Alpha blending / Optional
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; // Alpha blending / Optional
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

        VkPipelineColorBlendStateCreateInfo colorBlending = {};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f; // Optional
        colorBlending.blendConstants[1] = 0.0f; // Optional
        colorBlending.blendConstants[2] = 0.0f; // Optional
        colorBlending.blendConstants[3] = 0.0f; // Optional

        // 8. Dynamic States
        // A limited amount of the state that we've specified in the previous structs can actually be changed without recreating the pipeline. 
        // Examples are the size of the viewport, line width and blend constants.
        VkPipelineDynamicStateCreateInfo dynamicState = {};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        // 9. Pipeline layout
        // "uniform" values can be used in shaders, which are globals similar to dynamic state variables 
        // that can be changed at drawing time to alter the behavior of your shaders without having to recreate them. 
        // They are commonly used to pass the transformation matrix to the vertex shader, or to create texture samplers in the fragment shader. 
        // These uniform values need to be specified during pipeline creation by creating a VkPipelineLayout object.
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0; // optional
        pipelineLayoutInfo.pSetLayouts = nullptr; // optional

        // The structure also specifies push constants, which are another way of passing dynamic values to shaders
        pipelineLayoutInfo.pushConstantRangeCount = 0; // optional
        pipelineLayoutInfo.pPushConstantRanges = nullptr; // optional

        if(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Pipeline Layout!");
        }

        // 10. Graphics pipeline
        // ==> Shader stages: the shader modules that define the functionality of the programmable stages of the graphics pipeline
        // ==> Fixed-function state: all of the structures that define the fixed-function stages of the pipeline, like input assembly, rasterizer, viewport and color blending
        // ==> Pipeline layout: the uniform and push values referenced by the shader that can be updated at draw time
        // ==> Render pass: the attachments referenced by the pipeline stages and their usage
        // All of these combined fully define the functionality of the graphics pipeline
        VkGraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        
        // shader stages
        pipelineInfo.stageCount = 2; // 2 shader stages (vertex & fragment)
        pipelineInfo.pStages = shaderStages;
        
        // fixed-function stages
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pDepthStencilState = nullptr; // optional
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = nullptr; // optional

        pipelineInfo.layout = pipelineLayout;

        // Finally we have the reference to the render pass and the index of the sub pass 
        // where this graphics pipeline will be used. It is also possible to use other render passes 
        // with this pipeline instead of this specific instance, but they have to be compatible with "renderPass".
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;

        // Vulkan allows to create a new graphics pipeline by deriving from an existing pipeline. 
        // The idea of pipeline derivatives is that it is less expensive to set up pipelines when 
        // they have much functionality in common with an existing pipeline and switching 
        // between pipelines from the same parent can also be done quicker. 
        // We can either specify the handle of an existing pipeline with basePipelineHandle or 
        // reference another pipeline that is about to be created by index with basePipelineIndex. 
        // Right now there is only a single pipeline, so we'll simply specify a null handle and an invalid index. 
        // These values are only used if the VK_PIPELINE_CREATE_DERIVATIVE_BIT flag is also specified in the flags field of VkGraphicsPipelineCreateInfo.
        pipelineInfo.basePipelineIndex = -1; // optional
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // optional

        // Creation of the graphics pipeline
        if(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create graphics pipeline!");
        }

        // End of pipeline : destroy shader modules
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }
    //----------------------------------------------------------------------------------------------------------

    //----------------------------------------------------------------------------------------------
    //  FRAMEBUFFERS
    //----------------------------------------------------------------------------------------------
    
    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        // Create a framebuffer for each image view
        for(size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
                swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo = {};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass; // specifies with which renderpass this framebuffer needs to be compatible
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments; // specify the VkImageView objects that should be bound to the respective attachment descriptions in the render pass pAttachment array.
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create framebuffer!");
            }
        }
    }
    //----------------------------------------------------------------------------------------------

    //----------------------------------------------------------------------------------------------
    // COMMAND BUFFERS
    //----------------------------------------------------------------------------------------------
    
    // Command pools manage the memory that is used to store the buffers and 
    // command buffers are allocated from them.
    // Command buffers are executed by submitting them on one of the device queues, like the graphics 
    // and presentation queues we retrieved. Each command pool can only allocate command buffers that 
    // are submitted on a single type of queue. We're going to record commands for drawing, which is 
    // why we've chosen the graphics queue family.
    // There are two possible flags for command pools:
    // ==> VK_COMMAND_POOL_CREATE_TRANSIENT_BIT: Hint that command buffers are rerecorded with new commands very often (may change memory allocation behavior)
    // ==> VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: Allow command buffers to be rerecorded individually, without this flag they all have to be reset together
    // We will only record the command buffers at the beginning of the program and then execute them 
    // many times in the main loop, so we're not going to use either of these flags.
    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
        poolInfo.flags = 0; // optional

        if(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool!");
        }
    }

    void createCommandBuffers() {
        commandBuffers.resize(swapChainFramebuffers.size());

        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;

        // The level parameter specifies if the allocated command buffers are primary or secondary command buffers.
        // ==> VK_COMMAND_BUFFER_LEVEL_PRIMARY: Can be submitted to a queue for execution, but cannot be called from other command buffers.
        // ==> VK_COMMAND_BUFFER_LEVEL_SECONDARY: Cannot be submitted directly, but can be called from primary command buffers.
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

        if(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffers!");
        }

        // Starting command buffer recording
        for(size_t i = 0; i < commandBuffers.size(); i++) {
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            // The flags parameter specifies how we're going to use the command buffer. The following values are available:
            // ==> VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: The command buffer will be rerecorded right after executing it once.
            // ==> VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT: This is a secondary command buffer that will be entirely within a single render pass.
            // ==> VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT: The command buffer can be resubmitted while it is also already pending execution.
            beginInfo.flags = 0; // optional

            // Only relevant for secondary command buffers. It specifies which state to inherit from the calling primary command buffers.
            beginInfo.pInheritanceInfo = nullptr; // optional

            // If the command buffer was already recorded once, then a call to vkBeginCommandBuffer will implicitly reset it. 
            // It's not possible to append commands to a buffer at a later time.

            if(vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error("Failed to begin recording command buffer!");
            }

            // Starting a render pass
            // Drawing starts by beginning the render pass with vkCmdBeginRenderPass. 
            // The render pass is configured using some parameters in a VkRenderPassBeginInfo struct.
            // The first parameters are the render pass itself and the attachments to bind. 
            // We created a framebuffer for each swap chain image that specifies it as color attachment.
            VkRenderPassBeginInfo renderPassInfo = {};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = renderPass;
            renderPassInfo.framebuffer = swapChainFramebuffers[i];

            // The render area defines where shader loads and stores will take place. The pixels outside this region will have undefined values. 
            // It should match the size of the attachments for best performance.
            renderPassInfo.renderArea.offset = {0, 0};
            renderPassInfo.renderArea.extent = swapChainExtent;

            // The last two parameters define the clear values to use for VK_ATTACHMENT_LOAD_OP_CLEAR, 
            // which we used as load operation for the color attachment. 
            // I've defined the clear color to simply be black with 100% opacity.
            VkClearValue clearColor = {0.0f, 0.0f, 0.0f, 1.0f}; // black
            renderPassInfo.clearValueCount = 1;
            renderPassInfo.pClearValues = &clearColor;

            // Begin render pass
            //The render pass can now begin. All of the functions that record commands can be recognized by their vkCmd prefix. 
            // They all return void, so there will be no error handling until we've finished recording.
            // The first parameter for every command is always the command buffer to record the command to. 
            // The second parameter specifies the details of the render pass we've just provided. 
            // The final parameter controls how the drawing commands within the render pass will be provided. It can have one of two values:
            // ==> VK_SUBPASS_CONTENTS_INLINE: The render pass commands will be embedded in the primary command buffer itself and no secondary command buffers will be executed.
            // ==> VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS: The render pass commands will be executed from secondary command buffers.
            vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            // Bind graphics pipeline to command buffers
            // The second parameter specifies if the pipeline object is a graphics or compute pipeline.
            vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            // Drawing triangle
            // We've now told Vulkan which operations to execute in the graphics pipeline and 
            // which attachment to use in the fragment shader, so all that remains is telling it to draw the triangle.
            // It has the following parameters, aside from the command buffer:
            // ==> vertexCount: Even though we don't have a vertex buffer, we technically still have 3 vertices to draw.
            // ==> instanceCount: Used for instanced rendering, use 1 if you're not doing that.
            // ==> firstVertex: Used as an offset into the vertex buffer, defines the lowest value of gl_VertexIndex.
            // ==> firstInstance: Used as an offset for instanced rendering, defines the lowest value of gl_InstanceIndex.
            vkCmdDraw(commandBuffers[i], 3, 1, 0, 0);

            // End Render Pass
            vkCmdEndRenderPass(commandBuffers[i]);

            // Finish recording command buffer
            if(vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to record command buffer!");
            }
        }
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        // by default, fences are created in the unsignaled state. 
        // That means that vkWaitForFences will wait forever if we haven't used the fence before. 
        // To solve that, we can change the fence creation to initialize it in the signaled state as if we had rendered an initial frame that finished
        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        
        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS
            || vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS
            || vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create synchronization objects for a frame!");
            }
        }    
    }
    //----------------------------------------------------------------------------------------------

    //----------------------------------------------------------------------------------------------
    //  RECREATING SWAP CHAIN
    //----------------------------------------------------------------------------------------------

    void cleanupSwapChain() {
        // Destroy Framebuffers
        for(size_t i = 0; i < swapChainFramebuffers.size(); i++) {
            vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
        }

        // Deallocate/Free existing command buffers
        vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

        // Destroy graphics pipeline
        vkDestroyPipeline(device, graphicsPipeline, nullptr);

        // Destroy Pipeline layout
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        // Destroy Render Pass
        vkDestroyRenderPass(device, renderPass, nullptr);

        // Destroy Image Views
        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        // Destroy swap chain
        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    void recreateSwapChain() {
        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandBuffers();
    }
    //----------------------------------------------------------------------------------------------

    void initVulkan() {
        // setup
        createInstance();
        setupDebugMessenger(); // Debugging support (validation layers)
        createSurface(); // Window surface to present rendered images
        pickPhysicalDevice(); // select GPU (based on queue families, extensions and Swap Chain supports at least 1 feature (1 Surface format, 1 Present Mode))
        createLogicalDevice();

        // rendering infrastructure
        createSwapChain(); // (with surface format, present mode and swap extent)
        createImageViews(); // describes how to access the image and which part of the image to access, for example if it should be treated as a 2D texture depth texture without any mipmapping levels.
        createRenderPass(); // specify how many color and depth buffers there will be, how many samples to use for each of them and how their contents should be handled throughout the rendering operations. 
        createGraphicsPipeline();
        createFramebuffers(); // contained in swap chain

        // Drawing commands
        createCommandPool();
        createCommandBuffers();
        createSyncObjects(); // synchronize queue operations of draw commands and presentation
    }

    void mainLoop() {
        /*int keyState = glfwGetKey(window, GLFW_KEY_SPACE);
        while(keyState == GLFW_RELEASE) {
            glfwPollEvents();
            drawFrame();
        }*/                   
        if(!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        // All of the operations in drawFrame are asynchronous. That means that when we exit the loop in mainLoop, 
        // drawing and presentation operations may still be going on. Cleaning up resources while that is happening is a bad idea.
        // To fix that problem, we should wait for the logical device to finish operations before exiting mainLoop and destroying the window.
        vkDeviceWaitIdle(device);
    }

    void cleanup() {
        cleanupSwapChain();

        // Destroy Synchronization objects (semaphores and fences for each frame)
        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }
        
        // Destroy Command Pool
        vkDestroyCommandPool(device, commandPool, nullptr);

        // Destroy logical device
        vkDestroyDevice(device, nullptr);
        
        // if validation layers are enabled, destroy debug messenger
        if(enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        // Destroy window surface
        vkDestroySurfaceKHR(instance, surface, nullptr);

        // Destroy Vulkan instance
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    // The drawFrame function will perform the following operations:
    // ==> Acquire an image from the swap chain
    // ==> Execute the command buffer with that image as attachment in the framebuffer
    // ==> Return the image to the swap chain for presentation
    // Each of these events is set in motion using a single function call, 
    // but they are executed asynchronously. The function calls will return before the 
    // operations are actually finished and the order of execution is also undefined. 
    // That is unfortunate, because each of the operations depends on the previous one finishing.
    // There are two ways of synchronizing swap chain events: fences and semaphores. 
    // They're both objects that can be used for coordinating operations by having 
    // one operation signal and another operation wait for a fence or semaphore to go 
    // from the unsignaled to signaled state.
    // The difference is that the state of fences can be accessed from your program using calls 
    // like vkWaitForFences and semaphores cannot be. Fences are mainly designed to 
    // synchronize the application itself with rendering operation, 
    // whereas semaphores are used to synchronize operations within or across command queues. 
    // We want to synchronize the queue operations of draw commands and presentation, which makes semaphores the best fit.
    void drawFrame() {
        // Wait for the frame to be finished
        // The vkWaitForFences function takes an array of fences and waits for either any or all of them 
        // to be signaled before returning. The VK_TRUE we pass here indicates that we want to wait for all fences, 
        // but in the case of a single one it obviously doesn't matter. 
        // Just like vkAcquireNextImageKHR this function also takes a timeout. 
        // Unlike the semaphores, we manually need to restore the fence to the unsignaled state by resetting it with the vkResetFences call.
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        
        // 1. Acquire image from swap chain
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if(result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        } else if(result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("Failed to acquire swap chain image!");
        }
        // Check if a previous frame is using this image (there is its fence to wait on)
        if(imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
            vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        }

        // Mark the image as now being in use by this frame
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        // 2. Submitting the command buffer in the framebuffer
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        // The first three parameters specify which semaphores to wait on 
        // before execution begins and in which stage(s) of the pipeline to wait. 
        // We want to wait with writing colors to the image until it's available, 
        // so we're specifying the stage of the graphics pipeline that writes to the 
        // color attachment. That means that theoretically the implementation can already 
        // start executing our vertex shader and such while the image is not yet available. 
        // Each entry in the waitStages array corresponds to the semaphore with the same index in pWaitSemaphores.
        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        // The next two parameters specify which command buffers to actually submit for execution. 
        // As mentioned earlier, we should submit the command buffer that binds the swap chain image we just acquired as color attachment.
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

        // The signalSemaphoreCount and pSignalSemaphores parameters specify 
        // which semaphores to signal once the command buffer(s) have finished execution. 
        // In our case we're using the renderFinishedSemaphore for that purpose.
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        // The vkQueueSubmit call includes an optional parameter to pass a fence that should be signaled when the command buffer finishes executing. 
        // We can use this to signal that a frame has finished.
        if(vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        // 3. Presentation
        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores; // We wait until the render is finished and image returned to swap chain for presentation

        // Specify the swap chains to present images to and the index of the image for each swap chain. 
        // This will almost always be a single one.
        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        // There is one last optional parameter called pResults. 
        // It allows to specify an array of VkResult values to check for every individual swap chain 
        // if presentation was successful. It's not necessary if only a single swap chain is used, 
        // because we can simply use the return value of the present function.
        presentInfo.pResults = nullptr; // optional

        // Submits the request to present an image to the swap chain.
        result = vkQueuePresentKHR(presentQueue, &presentInfo);
        if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || frameBufferResized) {
            frameBufferResized = false;
            recreateSwapChain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to present swap chain image!");
        }

        // Advance to next frame
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

        // We may either get errors or notice that the memory usage slowly grows. 
        // The reason for this is that the application is rapidly submitting work in the drawFrame function, 
        // but doesn't actually check if any of it finishes. 
        // If the CPU is submitting work faster than the GPU can keep up with then the queue will slowly fill up with work. 
        // Worse, even, is that we are reusing the imageAvailableSemaphore and renderFinishedSemaphore semaphores, along with the command buffers, for multiple frames at the same time!
        vkQueueWaitIdle(presentQueue);
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}