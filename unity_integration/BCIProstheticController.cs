using UnityEngine;
using LSL;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// BCI Prosthetic Controller - Receives motor imagery commands via LSL
/// and controls the prosthetic hand animations in Unity.
/// </summary>
public class BCIProstheticController : MonoBehaviour
{
    [Header("LSL Settings")]
    [SerializeField] private string streamName = "ProstheticControl";
    [SerializeField] private float timeout = 1.0f;
    
    [Header("Hand Control")]
    [SerializeField] private Animator handAnimator;
    [SerializeField] private Transform wristTransform;
    [SerializeField] private float handSmoothSpeed = 2.0f;
    [SerializeField] private float wristSmoothSpeed = 2.0f;
    
    [Header("Animation Parameters")]
    [SerializeField] private string handOpenParam = "HandOpen";
    [SerializeField] private string wristRotationParam = "WristRotation";
    
    [Header("Debug")]
    [SerializeField] private bool showDebugInfo = true;
    
    // LSL variables
    private StreamInlet inlet;
    private float[] sample;
    private bool streamConnected = false;
    
    // State variables
    private float targetHandState = 0.5f;
    private float currentHandState = 0.5f;
    private float targetWristState = 0.5f;
    private float currentWristState = 0.5f;
    
    // Command tracking
    private int lastCommandType = 0;
    private float lastConfidence = 0.0f;
    
    void Start()
    {
        // Initialize sample buffer
        sample = new float[4]; // [hand_state, wrist_state, command_type, confidence]
        
        // Try to connect to LSL stream
        ConnectToLSL();
    }
    
    private void ConnectToLSL()
    {
        // Find LSL stream
        if (showDebugInfo) Debug.Log($"[BCI] Searching for LSL stream '{streamName}'...");
        
        var resolver = new StreamInlet.Resolver("name", streamName);
        
        StreamInfo[] streamInfos = LSL.LSL.resolve_stream("name", streamName, 1, timeout);
        
        if (streamInfos.Length > 0)
        {
            if (showDebugInfo) Debug.Log($"[BCI] Connected to LSL stream '{streamName}'!");
            
            // Create inlet to receive data
            inlet = new StreamInlet(streamInfos[0]);
            streamConnected = true;
        }
        else
        {
            if (showDebugInfo) Debug.LogWarning($"[BCI] No LSL stream named '{streamName}' found. Retrying...");
            Invoke(nameof(ConnectToLSL), 2.0f); // Retry after 2 seconds
        }
    }
    
    void Update()
    {
        if (streamConnected && inlet != null)
        {
            // Pull any available samples
            double timestamp = inlet.pull_sample(sample, 0.0);
            
            if (timestamp > 0)
            {
                // Parse the sample data
                float handState = sample[0];      // 0-1 (closed to open)
                float wristState = sample[1];     // 0-1 (left to right)
                int commandType = (int)sample[2]; // 0=idle, 1=hand, 2=wrist
                float confidence = sample[3];     // 0-1 confidence
                
                // Update target states
                targetHandState = handState;
                targetWristState = wristState;
                
                // Store for debugging
                lastCommandType = commandType;
                lastConfidence = confidence;
                
                if (showDebugInfo && commandType != 0)
                {
                    string cmdName = commandType == 1 ? "HAND" : "WRIST";
                    Debug.Log($"[BCI] Command: {cmdName}, Confidence: {confidence:F2}");
                }
            }
        }
        
        // Smooth transitions
        currentHandState = Mathf.Lerp(currentHandState, targetHandState, handSmoothSpeed * Time.deltaTime);
        currentWristState = Mathf.Lerp(currentWristState, targetWristState, wristSmoothSpeed * Time.deltaTime);
        
        // Apply animations
        UpdateHandAnimation();
        UpdateWristRotation();
    }
    
    void UpdateHandAnimation()
    {
        if (handAnimator != null)
        {
            // Set the hand open parameter (0 = closed fist, 1 = open hand)
            handAnimator.SetFloat(handOpenParam, currentHandState);
        }
        else
        {
            // Fallback: Scale-based animation if no animator
            if (transform != null)
            {
                Vector3 scale = transform.localScale;
                scale.x = 0.8f + currentHandState * 0.4f; // Scale from 0.8 to 1.2
                transform.localScale = scale;
            }
        }
    }
    
    void UpdateWristRotation()
    {
        if (wristTransform != null)
        {
            // Rotate wrist from -90 to +90 degrees based on state
            float rotationAngle = Mathf.Lerp(-90f, 90f, currentWristState);
            wristTransform.localRotation = Quaternion.Euler(0, rotationAngle, 0);
        }
    }
    
    void OnGUI()
    {
        if (showDebugInfo)
        {
            GUI.Box(new Rect(10, 10, 300, 150), "BCI Prosthetic Control");
            
            int y = 30;
            GUI.Label(new Rect(20, y, 280, 20), $"Stream: {(streamConnected ? "Connected" : "Disconnected")}");
            y += 25;
            
            GUI.Label(new Rect(20, y, 280, 20), $"Hand State: {currentHandState:F2} (Target: {targetHandState:F2})");
            y += 20;
            
            GUI.Label(new Rect(20, y, 280, 20), $"Wrist State: {currentWristState:F2} (Target: {targetWristState:F2})");
            y += 20;
            
            string cmdName = lastCommandType == 0 ? "IDLE" : (lastCommandType == 1 ? "HAND" : "WRIST");
            GUI.Label(new Rect(20, y, 280, 20), $"Last Command: {cmdName}");
            y += 20;
            
            GUI.Label(new Rect(20, y, 280, 20), $"Confidence: {lastConfidence:F2}");
        }
    }
    
    void OnDestroy()
    {
        if (inlet != null)
        {
            inlet.close_stream();
            Debug.Log("[BCI] LSL stream closed.");
        }
    }
} 