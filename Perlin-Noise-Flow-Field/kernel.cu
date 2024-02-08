#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <time.h>

#define _USE_MATH_DEFINES
#include <math.h>

#define _UTIL_TYPES
#define _UTIL_MATH
#define _UTIL_PERLIN_2D
#include "../utils.cuh"


__global__ void PerlinNoise(uint32_t *CudaMemory, float freq, int32_t depth)
{
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    float angle = Perlin2D(threadIdx.x, blockIdx.x, freq, depth);
    int8_t va = Map(angle, 0.0f, 1.0f, 0x00, 0xFF);

    CudaMemory[index] = (va << 16) + (va << 8) + va;
}

__global__ void MixParticles(vec2f_t *Particles, uint32_t Width, uint32_t Height)
{
    curandState state;
    curand_init(blockIdx.x * blockDim.x + threadIdx.x, 0, 0, &state);

    Particles[blockIdx.x * blockDim.x + threadIdx.x] = {
        curand_uniform(&state) * (Width - 1.0f),
        curand_uniform(&state) * (Height - 1.0f)
    };
}

__global__ void ClearScreen(color_t *CudaMemory, color_t color)
{
    CudaMemory[blockIdx.x * blockDim.x + threadIdx.x] = color;
}

__global__ void RenderParticles(vec2f_t *Particles, color_t *CudaMemory, uint32_t Width, uint32_t Height)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    vec2f_t v = Particles[idx];

    if ((int32_t)v.x < 0 || (int32_t)v.x >= Width || (int32_t)v.y < 0 || (int32_t)v.y >= Height) {
        curandState state;
        curand_init(idx, 0, 0, &state);

        Particles[idx] = {
            curand_uniform(&state) * (Width - 1.0f),
            curand_uniform(&state) * (Height - 1.0f)
        };
    }
    v = Particles[idx];

    float angle = Perlin2D(v.x, v.y, 0.012f, 10) * (M_PI * 2.0f);

    Particles[idx].x += cos(angle);
    Particles[idx].y += sin(angle);

    CudaMemory[(((int32_t)v.x) + (((int32_t)v.y) * Width))] = 0xFFFF0000;
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

int32_t main()
{
    ShowWindow(GetConsoleWindow(), SW_HIDE);

    HINSTANCE hInstance = GetModuleHandleW(NULL);

    WNDCLASSW wClass = { 0 };
    wClass.hbrBackground = (HBRUSH)COLOR_WINDOW;
    wClass.hInstance = hInstance;
    wClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    wClass.lpszClassName = L"Root";
    wClass.lpfnWndProc = WindowProc;

    if (!RegisterClassW(&wClass)) return -1;

    uint32_t Width = 1000;
    uint32_t Height = 1000;

    RECT window_rect = { 0 };
    window_rect.right = Width;
    window_rect.bottom = Height;
    window_rect.left = 0;
    window_rect.top = 0;

    AdjustWindowRect(&window_rect, WS_OVERLAPPEDWINDOW | WS_VISIBLE, 0);
    HWND window = CreateWindowW(
        wClass.lpszClassName,
        L"Perlin Noise Flow Fields",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        window_rect.right - window_rect.left,
        window_rect.bottom - window_rect.top,
        NULL, NULL,
        NULL, NULL
    );

    GetWindowRect(window, &window_rect);
    /* Not usefull for now
    uint32_t ClientWidth = window_rect.right - window_rect.left;
    uint32_t ClientHeight = window_rect.bottom - window_rect.top;
    */

    uint32_t BitmapWidth = Width;
    uint32_t BitmapHeight = Height;

    uint32_t BytesPerPixel = 4;

    void *memory = VirtualAlloc(
        0,
        BitmapWidth * BitmapHeight * BytesPerPixel,
        MEM_RESERVE | MEM_COMMIT,
        PAGE_READWRITE
    );

    vec2f_t *Particles;
    uint32_t Amount = 1000;
    uint32_t Multiplier = 100;
    cudaMalloc(&Particles, Multiplier * Amount * sizeof(vec2f_t));
    cudaDeviceSynchronize();

    MixParticles << <Multiplier, Amount >> > (Particles, BitmapWidth, BitmapHeight);
    cudaDeviceSynchronize();

    color_t* CudaMemory;
    cudaMalloc(&CudaMemory, BitmapWidth * BitmapHeight * BytesPerPixel);
    cudaDeviceSynchronize();
    
    BITMAPINFO BitmapInfo;
    BitmapInfo.bmiHeader.biSize = sizeof(BitmapInfo.bmiHeader);
    BitmapInfo.bmiHeader.biWidth = BitmapWidth;
    BitmapInfo.bmiHeader.biHeight = -BitmapHeight;
    BitmapInfo.bmiHeader.biPlanes = 1;
    BitmapInfo.bmiHeader.biBitCount = 32;
    BitmapInfo.bmiHeader.biCompression = BI_RGB;

    HDC hdc = GetDC(window);

    MSG msg = { 0 };
    int32_t running = 1;
    while (running) {
        while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) {
            switch (msg.message) {
            case WM_QUIT: {
                running = 0;
                break;
            }
            }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        
        ClearScreen << <BitmapHeight, BitmapWidth >> > (CudaMemory, 0xFF000000);
        cudaDeviceSynchronize();

        RenderParticles << <Multiplier, Amount >> > (Particles, CudaMemory, BitmapWidth, BitmapHeight);
        cudaDeviceSynchronize();

        cudaMemcpy(memory, CudaMemory, BitmapWidth * BitmapHeight * BytesPerPixel, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        StretchDIBits(
            hdc, 0, 0,
            BitmapWidth, BitmapHeight,
            0, 0,
            BitmapWidth, BitmapHeight,
            memory, &BitmapInfo,
            DIB_RGB_COLORS,
            SRCCOPY
        );
    }
    
    cudaFree(Particles);
    cudaFree(CudaMemory);
    VirtualFree(memory, 0, MEM_RELEASE);

    return 0;
}


LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg) {
        case WM_DESTROY:
            PostQuitMessage(0);
            break;
        default:
            return DefWindowProcW(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}