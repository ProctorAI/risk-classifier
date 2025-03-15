'use client';

import { useEffect, useState, useCallback, useRef } from "react";

interface EventData {
  type: string;
  data: Record<string, any>;
  timestamp: number;
  device_type?: string;
  screen_width?: number;
  screen_height?: number;
  window_width?: number;
  window_height?: number;
}

export default function useProctoring() {
  const [logs, setLogs] = useState<EventData[]>([]);
  const [throttledLogs, setThrottledLogs] = useState<EventData[]>([]);
  const [mousePositions, setMousePositions] = useState<{ x: number; y: number; timestamp: number }[]>([]);
  const [isFocused, setIsFocused] = useState(true); 
  const lastMouseMoveTimeRef = useRef(0); 
  const lastResizeTimeRef = useRef(0);
  const inactivityTimerRef = useRef<NodeJS.Timeout | null>(null); 
  const [deviceInfo, setDeviceInfo] = useState<{
    screenWidth: number;
    screenHeight: number;
    windowWidth: number;
    windowHeight: number;
    deviceType: string;
  } | null>(null);
  const MOUSE_THROTTLE = 500; // ms
  const RESIZE_THROTTLE = 500; // ms

  const logEvent = useCallback((type: string, data: Record<string, any> = {}) => {
    const eventData = { 
      type, 
      data, 
      timestamp: Date.now(),
      window_width: window.innerWidth,
      window_height: window.innerHeight
    };
    setLogs((prev) => [...prev, eventData]);
  }, []);

  // determine device type based on user agent
  const detectDeviceType = useCallback(() => {
    const userAgent = navigator.userAgent.toLowerCase();
    if (/mobile|android|iphone|ipad|ipod/.test(userAgent)) {
      return /ipad/.test(userAgent) ? "tablet" : "mobile";
    }
    return "desktop";
  }, []);

  // calculate normalized threshold for minimal movement
  const getNormalizedThreshold = useCallback(() => {
    if (deviceInfo) {
      const { screenWidth, screenHeight } = deviceInfo;
      const diagonal = Math.sqrt(screenWidth ** 2 + screenHeight ** 2);
      return 0.005 * diagonal; // 0.5% of screen diagonal as threshold
    }
    return 10; // default threshold if device info not yet available
  }, [deviceInfo]);

  useEffect(() => {
    const screenWidth = window.screen.width;
    const screenHeight = window.screen.height;
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;
    const deviceType = detectDeviceType();
    setDeviceInfo({ screenWidth, screenHeight, windowWidth, windowHeight, deviceType });
  }, [detectDeviceType]); 

  useEffect(() => {
    // mouse movement handler
    const handleMouseMove = (event: MouseEvent) => {
      const now = Date.now();
      if (now - lastMouseMoveTimeRef.current >= MOUSE_THROTTLE) {
        const { clientX: x, clientY: y } = event;
        logEvent("mouse_move", { x, y });
        setMousePositions((prev) => [...prev, { x, y, timestamp: now }]);
        lastMouseMoveTimeRef.current = now;

        // reset inactivity timer
        if (inactivityTimerRef.current) clearTimeout(inactivityTimerRef.current);
        inactivityTimerRef.current = setTimeout(() => logEvent("inactivity", { duration: 3000 }), 3000);
      }
    };

    // copy/paste handlers
    const handleCopy = (event: ClipboardEvent) => {
      logEvent("clipboard", { 
        action: "copy",
        selection: window.getSelection()?.toString()?.length || 0 // Only log length of copied text
      });
    };

    const handlePaste = (event: ClipboardEvent) => {
      logEvent("clipboard", { 
        action: "paste",
        length: event.clipboardData?.getData('text')?.length || 0 // Only log length of pasted text
      });
    };

    const handleCut = (event: ClipboardEvent) => {
      logEvent("clipboard", { 
        action: "cut",
        selection: window.getSelection()?.toString()?.length || 0 // Only log length of cut text
      });
    };

    // window resize handler
    const handleResize = () => {
      const now = Date.now();
      if (now - lastResizeTimeRef.current >= RESIZE_THROTTLE) {
        logEvent("window_resize", {
          width: window.innerWidth,
          height: window.innerHeight,
          ratio: window.innerWidth / window.screen.width
        });
        lastResizeTimeRef.current = now;
      }
    };

    // keystroke handler
    const handleKeyDown = (event: KeyboardEvent) => {
      const keyType = event.key.length === 1 ? "character" : event.key;
      logEvent("key_press", { key_type: keyType });
      if (inactivityTimerRef.current) clearTimeout(inactivityTimerRef.current);
      inactivityTimerRef.current = setTimeout(() => logEvent("inactivity", { duration: 3000 }), 3000);
    };

    // tab switch handler
    const handleVisibilityChange = () => {
      logEvent("tab_switch", { status: document.hidden ? "hidden" : "visible" });
    };

    // window focus handlers
    const handleFocus = () => {
      if (!isFocused) {
        setIsFocused(true);
        logEvent("window_state_change", { state: "focused" });
      }
    };

    const handleBlur = () => {
      if (isFocused) {
        setIsFocused(false);
        logEvent("window_state_change", { state: "blurred" });
      }
    };

    // process logs every 5 seconds
    const interval = setInterval(() => {
      if (logs.length > 0) {
        const windowStart = Date.now() - 5000;
        const recentMouseData = mousePositions.filter((pos) => pos.timestamp >= windowStart);

        // analyze mouse movement (compare positions at 2s and 5s)
        if (recentMouseData.length >= 2) {
          const second2Index = recentMouseData.findIndex((pos) => pos.timestamp >= windowStart + 2000);
          const second2 = second2Index >= 0 ? recentMouseData[second2Index] : recentMouseData[0];
          const second5 = recentMouseData[recentMouseData.length - 1];
          const distance = Math.sqrt(
            Math.pow(second5.x - second2.x, 2) + Math.pow(second5.y - second2.y, 2)
          );
          const normalizedThreshold = getNormalizedThreshold();
          logEvent("mouse_analysis", { distance, minimalMovement: distance < normalizedThreshold });
        }

        logEvent("focus_state", { isFocused });

        const logsWithDeviceInfo = logs.map(log => ({
          ...log,
          device_type: deviceInfo?.deviceType,
          screen_width: deviceInfo?.screenWidth,
          screen_height: deviceInfo?.screenHeight,
          window_width: deviceInfo?.windowWidth,
          window_height: deviceInfo?.windowHeight
        }));
        setThrottledLogs([...logsWithDeviceInfo]);
        setLogs([]);
        setMousePositions((prev) => prev.filter((pos) => pos.timestamp >= windowStart));
      }
    }, 5000);

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("keydown", handleKeyDown);
    document.addEventListener("visibilitychange", handleVisibilityChange);
    document.addEventListener("copy", handleCopy);
    document.addEventListener("paste", handlePaste);
    document.addEventListener("cut", handleCut);
    window.addEventListener("resize", handleResize);
    window.addEventListener("focus", handleFocus);
    window.addEventListener("blur", handleBlur);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      document.removeEventListener("copy", handleCopy);
      document.removeEventListener("paste", handlePaste);
      document.removeEventListener("cut", handleCut);
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("focus", handleFocus);
      window.removeEventListener("blur", handleBlur);
      clearInterval(interval);
      if (inactivityTimerRef.current) clearTimeout(inactivityTimerRef.current);
    };
  }, [logEvent, getNormalizedThreshold, isFocused]); 

  return throttledLogs;
}