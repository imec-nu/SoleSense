#include <driver/adc.h>
#include <esp_adc/adc_continuous.h>
#include <esp_log.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>

int file_counter = 0;  

#define SAMPLES_PER_SECOND 16000   // 16000
#define READ_LEN 1024              //
#define NUM_CHANNELS 3             // 
#define SAMPLE_BITS 16
#define BUFFER_SIZE 160   //

static adc_channel_t channel[NUM_CHANNELS] = {
    ADC_CHANNEL_0,  //
    ADC_CHANNEL_1,  // 
    ADC_CHANNEL_2  //
};
static TaskHandle_t s_task_handle;
static SemaphoreHandle_t dma_semaphore;
adc_continuous_handle_t handle = NULL;
unsigned long start_time = 0;  // 
unsigned long end_time = 0;    // 
// 

uint16_t bufferA[NUM_CHANNELS][BUFFER_SIZE];  
uint16_t bufferB[NUM_CHANNELS][BUFFER_SIZE];  
uint16_t sample_index1 = 0, sample_index2 = 0, sample_index3 = 0;  // 
volatile bool bufferAReady = false;
volatile bool bufferBReady = false;
volatile bool usingBufferA = true;  //
SemaphoreHandle_t bufferSemaphore;  //



void send_data_task(void *arg) {
    Serial.printf("Send data task running on core %d\n", xPortGetCoreID());
    // uint8_t frame_header[2] = {0xAA, 0x55};
    uint8_t frame_header[2] = {0xCC, 0x44};
    while (1) {
        // 
        xSemaphoreTake(bufferSemaphore, portMAX_DELAY);

        if (bufferAReady) {
            Serial.write(frame_header, 2);  // 
            Serial.write((uint8_t*)bufferA, BUFFER_SIZE * NUM_CHANNELS * sizeof(uint16_t));
            bufferAReady = false;
        }
        if (bufferBReady) {
            Serial.write(frame_header, 2); 
            Serial.write((uint8_t*)bufferB, BUFFER_SIZE * NUM_CHANNELS * sizeof(uint16_t));
            bufferBReady = false;
        }
    }
}


//
static bool IRAM_ATTR adcComplete(adc_continuous_handle_t handle, const adc_continuous_evt_data_t *edata, void *user_data) {
    BaseType_t mustYield = pdFALSE;
    vTaskNotifyGiveFromISR(s_task_handle, &mustYield);
    return (mustYield == pdTRUE);
}

static void adc_init(adc_channel_t *channel, uint8_t channel_num) {
    adc_continuous_handle_cfg_t adc_config = {
        .max_store_buf_size = 8192,   // 
        .conv_frame_size = READ_LEN,  //
    };
    ESP_ERROR_CHECK(adc_continuous_new_handle(&adc_config, &handle));
    
    adc_continuous_config_t dig_cfg = {
        .sample_freq_hz = SAMPLES_PER_SECOND * NUM_CHANNELS,  
        .conv_mode = ADC_CONV_SINGLE_UNIT_1,  
        .format = ADC_DIGI_OUTPUT_FORMAT_TYPE2, 
    };
    adc_digi_pattern_config_t adc_pattern[NUM_CHANNELS] = {0};
    dig_cfg.pattern_num = channel_num;
    for (int i = 0; i < channel_num; i++) {
        adc_pattern[i].atten = ADC_ATTEN_DB_11;
        adc_pattern[i].channel = channel[i];
        adc_pattern[i].unit = ADC_UNIT_1;
        adc_pattern[i].bit_width = SOC_ADC_DIGI_MAX_BITWIDTH;
    }
    dig_cfg.adc_pattern = adc_pattern;
    ESP_ERROR_CHECK(adc_continuous_config(handle, &dig_cfg));
}


void stop_adc_and_clear_dma() {

    ESP_ERROR_CHECK(adc_continuous_stop(handle));

    uint8_t temp_buffer[READ_LEN];
    uint32_t ret_num = 0;
}

void restart_adc() {
    ESP_ERROR_CHECK(adc_continuous_start(handle));
}

void setup() {
    Serial.begin(115200);


    s_task_handle = xTaskGetCurrentTaskHandle();
    dma_semaphore = xSemaphoreCreateBinary();
   
    bufferSemaphore = xSemaphoreCreateBinary();


    adc_init(channel, NUM_CHANNELS);


    adc_continuous_evt_cbs_t cbs = {
        .on_conv_done = adcComplete,  
    };
    ESP_ERROR_CHECK(adc_continuous_register_event_callbacks(handle, &cbs, NULL));

    
    ESP_ERROR_CHECK(adc_continuous_start(handle));

    
    xTaskCreatePinnedToCore(send_data_task, "SendDataTask", 4096, NULL, 1, NULL, 1);

    start_time = micros();  
}

void loop() {
    uint8_t result[READ_LEN] = {0};  
    uint32_t ret_num = 0;
    while (1) {
        
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        
        while (adc_continuous_read(handle, result, READ_LEN, &ret_num, 0) == ESP_OK) {
            for (int i = 0; i < ret_num; i += SOC_ADC_DIGI_RESULT_BYTES) {
                adc_digi_output_data_t *p = (adc_digi_output_data_t *)&result[i];
                
               if (usingBufferA) {
                    switch (p->type2.channel) {
                        case ADC_CHANNEL_0: bufferA[0][sample_index1++] = p->type2.data; break;
                        case ADC_CHANNEL_1: bufferA[1][sample_index2++] = p->type2.data; break;
                        case ADC_CHANNEL_2: bufferA[2][sample_index3++] = p->type2.data; break;
                        
                    }
                } else {
                    switch (p->type2.channel) {
                        case ADC_CHANNEL_0: bufferB[0][sample_index1++] = p->type2.data; break;
                        case ADC_CHANNEL_1: bufferB[1][sample_index2++] = p->type2.data; break;
                        case ADC_CHANNEL_2: bufferB[2][sample_index3++] = p->type2.data; break;
                       
                    }
                }

                
                if (sample_index1 >= BUFFER_SIZE && sample_index2 >= BUFFER_SIZE &&
                    sample_index3 >= BUFFER_SIZE ) {

                    
                    if (usingBufferA) {
                        bufferAReady = true;
                    } else {
                        bufferBReady = true;
                    }
                    sample_index1 = sample_index2 = sample_index3 = 0;
                    usingBufferA = !usingBufferA;  
                    xSemaphoreGive(bufferSemaphore);  

                }
            }
        }
    }
}