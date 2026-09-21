import React, { useState, useEffect, useRef, useMemo } from 'react';
import DeckGL from '@deck.gl/react';
import { ColumnLayer } from '@deck.gl/layers';
import { OrbitView, COORDINATE_SYSTEM } from '@deck.gl/core';

const INITIAL_VIEW_STATE = {
  target: [0, 0, 0],
  zoom: 1, // 1 is often good for cartesian meters around 100m range
  rotationX: 60,
  rotationOrbit: 180, // Look forward
};

const COLOR_MAP = {
  0: [128, 128, 128], // Terrain (Gray)
  1: [50, 100, 255],  // Static (Blue)
  2: [255, 50, 50],   // Dynamic (Red)
};

function polarToCartesian(ring_idx, angle_idx, res) {
  const range = ring_idx * res + res / 2;
  const angular_step = res / Math.max(range, 0.5);
  const theta = angle_idx * angular_step;
  const x = range * Math.cos(theta);
  const y = range * Math.sin(theta);
  return [x, y];
}

export default function App() {
  const [frameData, setFrameData] = useState(null);
  const [speed, setSpeed] = useState(8.0);
  const [steering, setSteering] = useState(0.0);

  useEffect(() => {
    let ws = null;
    let reconnectTimeout = null;

    function connect() {
      ws = new WebSocket('ws://localhost:8000/ws/grid');
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setFrameData(data);
        } catch (err) {
          console.error("Failed to parse websocket frame:", err);
        }
      };

      ws.onclose = () => {
        // Attempt reconnect after 1 second
        reconnectTimeout = setTimeout(connect, 1000);
      };
    }
    
    connect();

    return () => {
      clearTimeout(reconnectTimeout);
      if (ws) ws.close();
    };
  }, []);

  const handleVehicleUpdate = async (newSpeed, newSteering) => {
    setSpeed(newSpeed);
    setSteering(newSteering);
    try {
      await fetch('http://localhost:8000/vehicle_state', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          speed_mps: newSpeed,
          steering_angle_rad: newSteering
        })
      });
    } catch (e) {
      console.warn('Failed to POST vehicle_state', e);
    }
  };

  const layers = useMemo(() => {
    if (!frameData || !frameData.cells) return [];

    const groundData = [];
    const obstacleData = [];

    frameData.cells.forEach(cell => {
      const [x, y] = polarToCartesian(cell.ring_idx, cell.angle_idx, cell.resolution_tier);
      const color = COLOR_MAP[cell.semantic_class] || [255, 255, 255];
      const gndZ = cell.elevation_ground;

      // Render the ground cell (giving it a small 0.1m thickness for visibility)
      groundData.push({
        pos: [x, y, gndZ - 0.1],
        height: 0.1,
        radius: cell.resolution_tier / 2,
        color: color
      });

      // Render the obstacle top if present
      if (cell.elevation_obstacle_top !== null) {
        const obsBot = cell.elevation_obstacle_bottom !== null ? cell.elevation_obstacle_bottom : gndZ;
        const height = Math.max(0.1, cell.elevation_obstacle_top - obsBot);
        
        obstacleData.push({
          pos: [x, y, obsBot],
          height: height,
          radius: cell.resolution_tier / 2,
          color: [...color, 180] // Semi-transparent
        });
      }
    });

    return [
      new ColumnLayer({
        id: 'ground-layer',
        data: groundData,
        diskResolution: 6,
        radius: d => d.radius,
        extruded: true,
        getPosition: d => d.pos,
        getElevation: d => d.height,
        getFillColor: d => d.color,
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        material: false
      }),
      new ColumnLayer({
        id: 'obstacle-layer',
        data: obstacleData,
        diskResolution: 6,
        radius: d => d.radius,
        extruded: true,
        getPosition: d => d.pos,
        getElevation: d => d.height,
        getFillColor: d => d.color,
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        material: false
      })
    ];
  }, [frameData]);

  return (
    <div style={styles.container}>
      <DeckGL
        views={new OrbitView({ id: 'orbit', controller: true })}
        initialViewState={INITIAL_VIEW_STATE}
        layers={layers}
        parameters={{
          depthTest: true,
          blendFunc: ['SRC_ALPHA', 'ONE_MINUS_SRC_ALPHA']
        }}
      />
      
      {/* Controls Overlay */}
      <div style={styles.controlPanel}>
        <h3 style={{margin: '0 0 10px 0'}}>Vehicle Kinematics</h3>
        
        <div style={styles.inputRow}>
          <label style={styles.label}>Speed ({speed.toFixed(1)} m/s)</label>
          <input 
            type="range" min="0" max="30" step="0.5" 
            value={speed} 
            onChange={(e) => handleVehicleUpdate(parseFloat(e.target.value), steering)}
          />
        </div>
        
        <div style={styles.inputRow}>
          <label style={styles.label}>Steering ({steering.toFixed(2)} rad)</label>
          <input 
            type="range" min="-1" max="1" step="0.05" 
            value={steering} 
            onChange={(e) => handleVehicleUpdate(speed, parseFloat(e.target.value))}
          />
        </div>
      </div>

      {/* Metrics Panel */}
      <div style={styles.statsPanel}>
        <h3 style={{margin: '0 0 10px 0'}}>Telemetry</h3>
        {frameData ? (
          <div style={styles.statsGrid}>
            <div>FPS:</div>
            <div>{frameData.metrics.fps.toFixed(1)}</div>
            
            <div>Memory Savings:</div>
            <div>{frameData.metrics.memory_savings_pct.toFixed(1)}%</div>
            
            <div>Grid Cells:</div>
            <div>{frameData.cells.length.toLocaleString()}</div>
            
            <div>Foveated Size:</div>
            <div>{(frameData.metrics.memory_bytes_foveated / 1024).toFixed(1)} KB</div>
          </div>
        ) : (
          <div style={{color: '#aaa', fontStyle: 'italic'}}>Waiting for stream...</div>
        )}
      </div>
    </div>
  );
}

// Inline styling for prototype
const styles = {
  container: {
    position: 'absolute',
    top: 0, left: 0, right: 0, bottom: 0,
    backgroundColor: '#1a1a1a',
    fontFamily: 'system-ui, -apple-system, sans-serif'
  },
  controlPanel: {
    position: 'absolute',
    bottom: 30,
    left: 30,
    backgroundColor: 'rgba(20, 25, 30, 0.85)',
    color: '#e0e0e0',
    padding: '20px',
    borderRadius: '12px',
    boxShadow: '0 4px 15px rgba(0,0,0,0.5)',
    border: '1px solid rgba(255,255,255,0.1)',
    backdropFilter: 'blur(4px)',
    width: '280px',
  },
  inputRow: {
    display: 'flex',
    flexDirection: 'column',
    marginBottom: '15px'
  },
  label: {
    marginBottom: '5px',
    fontSize: '0.9rem',
    fontWeight: '500'
  },
  statsPanel: {
    position: 'absolute',
    top: 30,
    right: 30,
    backgroundColor: 'rgba(20, 25, 30, 0.85)',
    color: '#e0e0e0',
    padding: '20px',
    borderRadius: '12px',
    boxShadow: '0 4px 15px rgba(0,0,0,0.5)',
    border: '1px solid rgba(255,255,255,0.1)',
    backdropFilter: 'blur(4px)',
    width: '250px',
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr auto',
    gap: '8px 15px',
    fontSize: '0.95rem'
  }
};
