# macOS LSL Setup Guide

## Problem
The error `DllNotFoundException: lsl` occurs because Unity cannot find the Lab Streaming Layer (LSL) native library on macOS. The native library needs to be added as a Unity plugin.

## Solution: Add LSL as Native Plugin

### Step 1: Copy LSL Library to Unity Project
1. **Create the Plugins directory** in your Unity project:
   ```
   Assets/Plugins/macOS/
   ```

2. **Copy the LSL library** from your `lib/` directory:
   ```bash
   # Copy the main LSL library
   cp /path/to/your/project/lib/liblsl.dylib /path/to/Unity/Assets/Plugins/macOS/
   ```

### Step 2: Configure Plugin Settings in Unity
1. **Select the library** in Unity Project window:
   - Navigate to `Assets/Plugins/macOS/liblsl.dylib`
   - Click on it to select

2. **Configure in Inspector**:
   - **Platform settings**: ✅ macOS
   - **Settings for macOS**:
     - SDK: Any SDK
     - CPU: Any CPU (or Intel64/Apple Silicon as needed)
     - PlaceholderPath: `liblsl.dylib`
     - Load on startup: ✅
   - **Apply** the settings

### Step 3: Verify LSL C# Scripts
Make sure you're using the standard LSL C# wrapper, not iOS-specific versions:

1. **Use `BCIProstheticController.cs`** (not the iOS version)
2. **Ensure proper LSL imports**:
   ```csharp
   using LSL;  // Make sure this namespace is available
   ```

### Step 4: Test the Setup

1. **Build and run** your Unity project
2. **Check for LSL streams**:
   ```csharp
   // In your Unity script
   try {
       StreamInfo[] results = LSL.resolve_stream("name", "ProstheticControl", 1, 1.0);
       Debug.Log($"Found {results.Length} LSL streams");
   } catch (DllNotFoundException e) {
       Debug.LogError($"LSL library not found: {e.Message}");
   }
   ```

## Alternative: Direct Library Copy

If the plugin approach doesn't work, you can also copy the library directly to the built app:

```bash
# After building your Unity app
cp lib/liblsl.dylib "/path/to/built/app.app/Contents/MacOS/"
```

## Common Issues

### Issue: Library Architecture Mismatch
If you get architecture errors:
- Make sure the `liblsl.dylib` matches your Mac's architecture (Intel vs Apple Silicon)
- You may need to build LSL specifically for your architecture

### Issue: Code Signing Problems
If macOS blocks the library:
```bash
# Remove quarantine attribute
xattr -d com.apple.quarantine /path/to/liblsl.dylib
```

### Issue: Unity Still Can't Find Library
Try renaming in Unity:
- Use filename: `lsl.bundle` instead of `liblsl.dylib`
- Or use: `lsl.dylib`

## Verification Commands

Test that LSL is working from command line:
```bash
# Check if LSL library is linked correctly
otool -L lib/liblsl.dylib

# Test LSL from Python (should work)
python -c "import pylsl; print('LSL version:', pylsl.version())"
```

## Success Confirmation

When working correctly, you should see in Unity console:
```
[BCI] LSL stream 'ProstheticControl' found!
[BCI] Connected to stream with 4 channels
```

Instead of:
```
DllNotFoundException: lsl
``` 