using UnityEngine;
using LSL;

/// <summary>
/// Optional Unity script that sends a ready signal to the BCI system.
/// This allows the BCI system to detect when Unity is fully loaded and ready.
/// </summary>
public class UnityReadySignal : MonoBehaviour
{
    [Header("LSL Settings")]
    [SerializeField] private bool sendReadySignal = true;
    [SerializeField] private float signalDuration = 5.0f; // How long to maintain the signal
    
    private StreamOutlet readyOutlet;
    private float signalStartTime;
    private bool signalSent = false;
    
    void Start()
    {
        if (sendReadySignal)
        {
            CreateReadyStream();
        }
    }
    
    void CreateReadyStream()
    {
        try
        {
            // Create a simple ready signal stream
            StreamInfo readyInfo = new StreamInfo(
                "UnityReady",           // Stream name that BCI looks for
                "Status",               // Stream type
                1,                      // Channel count
                0,                      // Irregular sampling rate
                channel_format_t.cf_float32,
                "UnityProstheticApp"    // Source ID
            );
            
            // Create outlet
            readyOutlet = new StreamOutlet(readyInfo);
            signalStartTime = Time.time;
            
            Debug.Log("[UnityReady] Created Unity ready signal stream");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[UnityReady] Failed to create ready stream: {e.Message}");
        }
    }
    
    void Update()
    {
        // Send ready signal periodically for the specified duration
        if (sendReadySignal && readyOutlet != null && !signalSent)
        {
            if (Time.time - signalStartTime < signalDuration)
            {
                // Send a "1" to indicate Unity is ready
                float[] sample = { 1.0f };
                readyOutlet.push_sample(sample);
                
                // Log first signal
                if (Time.time - signalStartTime < 0.1f)
                {
                    Debug.Log("[UnityReady] Sending ready signal to BCI system...");
                }
            }
            else
            {
                // Signal duration complete
                signalSent = true;
                Debug.Log("[UnityReady] Ready signal transmission complete");
                
                // Optionally destroy the outlet after signaling
                if (readyOutlet != null)
                {
                    // Note: LSL4Unity doesn't have explicit outlet destruction
                    readyOutlet = null;
                }
            }
        }
    }
    
    void OnApplicationQuit()
    {
        if (readyOutlet != null)
        {
            // Send a "0" to indicate Unity is shutting down
            float[] sample = { 0.0f };
            readyOutlet.push_sample(sample);
            Debug.Log("[UnityReady] Sent shutdown signal");
        }
    }
} 