from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
try:
    # Python 3.7+ provides ThreadingHTTPServer
    from http.server import ThreadingHTTPServer as _ThreadingHTTPServer  # type: ignore
except Exception:  # pragma: no cover
    _ThreadingHTTPServer = HTTPServer  # type: ignore
import threading
from urllib.parse import urlparse, parse_qs
import time
from typing import Callable, Dict, Any, Optional

from .logging_setup import init_logger


class TelemetryServer:
    def __init__(
        self,
        addr: str,
        port: int,
        get_status: Callable[[], Dict[str, Any]],
        on_heartbeat: Optional[Callable[[], None]] = None,
        on_set_mode: Optional[Callable[[str], None]] = None,
        on_manual: Optional[Callable[..., None]] = None,
        on_pan_tilt: Optional[Callable[[Optional[int], Optional[int], Optional[int], Optional[int], Optional[bool]], None]] = None,
        get_snapshot: Optional[Callable[[], Optional[bytes]]] = None,
        get_map: Optional[Callable[[], Optional[bytes]]] = None,
        on_slam: Optional[Callable[[str, Dict[str, Any]], Any]] = None,  # New SLAM callback
        logger_name: str = "picarx.telemetry",
    ) -> None:
        self.log = init_logger(logger_name)
        self._addr = addr
        self._port = int(port)
        self._get_status = get_status
        self._on_heartbeat = on_heartbeat
        self._on_set_mode = on_set_mode
        self._on_manual = on_manual
        self._on_pan_tilt = on_pan_tilt
        self._get_snapshot = get_snapshot
        self._get_map = get_map
        self._on_slam = on_slam  # Store SLAM callback
        self._srv = None
        self._thread = None

    def start(self):
        if self._srv is not None:
            return

        get_status = self._get_status
        on_beat = self._on_heartbeat
        on_mode = self._on_set_mode
        on_manual = self._on_manual
        on_cam = self._on_pan_tilt
        get_snapshot = self._get_snapshot
        on_slam = self._on_slam
        get_map = self._get_map
        log = self.log

        class Handler(BaseHTTPRequestHandler):
            def _serve_index(self):
                html = (
                    "<html><head><meta name='viewport' content='width=device-width, initial-scale=1'>"
                    "<meta charset='UTF-8'>"  # Add charset for proper character encoding
                    "<style>"
                    "body{font-family:sans-serif;margin:1rem;background:#f5f5f5}"
                    ".card{background:white;border-radius:8px;padding:1rem;margin:0.5rem 0;box-shadow:0 2px 4px rgba(0,0,0,0.1)}"
                    ".header{background:#2c3e50;color:white;padding:1rem;border-radius:8px;margin-bottom:1rem}"
                    ".status-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem;margin:1rem 0}"
                    ".metric{text-align:center;padding:0.5rem;background:#ecf0f1;border-radius:4px}"
                    ".metric-value{font-size:1.5em;font-weight:bold;margin:0.25rem 0}"
                    ".metric-label{font-size:0.8em;color:#7f8c8d}"
                    ".danger{background:#e74c3c;color:white} .caution{background:#f39c12;color:white} .safe{background:#27ae60;color:white}"
                    ".btn-group{display:flex;gap:0.5rem;flex-wrap:wrap} button{padding:0.5rem 1rem;border:none;border-radius:4px;cursor:pointer;background:#3498db;color:white}"
                    "button:hover{background:#2980b9} .btn-danger{background:#e74c3c} .btn-warning{background:#f39c12} .btn-success{background:#27ae60}"
                    ".collapsible{cursor:pointer;padding:0.5rem;background:#34495e;color:white;border:none;width:100%;text-align:left;border-radius:4px;margin:0.25rem 0}"
                    ".collapsible:hover{background:#2c3e50} .content{display:none;padding:1rem;background:#ecf0f1;border-radius:0 0 4px 4px}"
                    ".content.active{display:block} .camera-container{max-width:400px} .camera-stream{width:100%;border-radius:4px}"
                    "</style>"
                    "</head><body>"
                    "<div class='header'><h2 style='margin:0'>Picar-X Control Dashboard</h2></div>"
                    
                    # Essential Status Overview
                    "<div class='card'>"
                    "<h3>System Status</h3>"
                    "<div class='status-grid'>"
                    "<div class='metric'><div class='metric-value' id='mode-value'>--</div><div class='metric-label'>Mode</div></div>"
                    "<div class='metric'><div class='metric-value' id='distance-value'>--</div><div class='metric-label'>Distance (cm)</div></div>"
                    "<div class='metric'><div class='metric-value' id='battery-value'>--</div><div class='metric-label'>Battery (V)</div></div>"
                    "<div class='metric' id='safety-metric'><div class='metric-value' id='safety-value'>--</div><div class='metric-label'>Safety Status</div></div>"
                    "</div>"
                    "</div>"
                    
                    # Primary Controls
                    "<div class='card'>"
                    "<h3>Primary Controls</h3>"
                    "<div class='btn-group'>"
                    "<button onclick=fetch('/mode?m=manual')>Manual</button>"
                    "<button onclick=fetch('/mode?m=auto')>Autonomous</button>"
                    "<button onclick=fetch('/mode?m=idle') class='btn-danger'>Stop</button>"
                    "</div>"
                    "</div>"
                    
                    # Camera Feed
                    "<div class='card'>"
                    "<h3>Camera Feed</h3>"
                    "<div class='camera-container'>"
                    "<img id='stream' class='camera-stream' onerror='this.style.border=\"3px solid red\";this.alt=\"Stream Error\"' onload='this.style.border=\"3px solid green\"'/>"
                    "<div class='btn-group' style='margin-top:0.5rem'>"
                    "<button id='openCam'>Open Full View</button>"
                    "<button id='refreshCam'>Refresh Stream</button>"
                    "</div>"
                    "</div>"
                    "</div>"
                    
                    # Collapsible Sections
                    "<button class='collapsible' onclick='toggleSection(this)'>Manual Controls</button>"
                    "<div class='content'>"
                    "<div class='btn-group'>"
                    "<button onclick=fetch('/manual?speed=50')>Forward</button>"
                    "<button onclick=fetch('/manual?speed=-50')>Backward</button>"
                    "<button onclick=fetch('/manual?speed=0')>Stop</button>"
                    "</div>"
                    "<div class='btn-group' style='margin-top:0.5rem'>"
                    "<button onclick=fetch('/manual?steer_delta=-5')>Turn Left</button>"
                    "<button onclick=fetch('/manual?steer=0')>Center</button>"
                    "<button onclick=fetch('/manual?steer_delta=5')>Turn Right</button>"
                    "</div>"
                    "<small style='display:block;margin-top:0.5rem;color:#7f8c8d'>Use Turn Left/Right for incremental steering adjustments</small>"
                    "</div>"
                    
                    "<button class='collapsible' onclick='toggleSection(this)'>Camera Controls</button>"
                    "<div class='content'>"
                    "<div class='btn-group'>"
                    "<button onclick=fetch('/camera?pan_delta=-5')>Pan Left</button>"
                    "<button onclick=fetch('/camera?pan=0&tilt=0')>Center Camera</button>"
                    "<button onclick=fetch('/camera?pan_delta=5')>Pan Right</button>"
                    "</div>"
                    "<div class='btn-group' style='margin-top:0.5rem'>"
                    "<button onclick=fetch('/camera?tilt_delta=5')>Tilt Up</button>"
                    "<button onclick=fetch('/camera?tilt_delta=-5')>Tilt Down</button>"
                    "</div>"
                    "</div>"
                    
                    "<button class='collapsible' onclick='toggleSection(this)'>Enhanced SLAM & Navigation</button>"
                    "<div class='content'>"
                    "<div class='metrics-grid' style='display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin-bottom:1rem'>"
                    "<div class='metric-card'>"
                    "<h4>Map Status</h4>"
                    "<div>Mode: <span id='slam-mode'>exploration</span></div>"
                    "<div>Pose: (<span id='slam-x'>0.0</span>, <span id='slam-y'>0.0</span>)</div>"
                    "<div>Confidence: <span id='slam-confidence'>100%</span></div>"
                    "<div style='margin-top:0.5rem'>"
                    "<button onclick='slam.setHome()' style='font-size:0.8em;padding:0.2rem 0.5rem'>🏠 Set Home</button>"
                    "<button onclick='slam.clearMap()' style='font-size:0.8em;padding:0.2rem 0.5rem;margin-left:0.3rem'>🗑️ Clear</button>"
                    "</div>"
                    "</div>"
                    "<div class='metric-card'>"
                    "<h4>Exploration</h4>"
                    "<div>Area: <span id='slam-area'>0.0</span> m²</div>"
                    "<div>Distance: <span id='slam-distance'>0.0</span> m</div>"
                    "<div>Features: <span id='slam-features'>0</span></div>"
                    "<div>Obstacles: <span id='slam-obstacles'>0</span></div>"
                    "</div>"
                    "<div class='metric-card'>"
                    "<h4>Navigation</h4>"
                    "<div>Goal: <span id='slam-goal'>None</span></div>"
                    "<div>Home: <span id='slam-home'>None</span></div>"
                    "<div>Path: <span id='slam-path'>0</span> steps</div>"
                    "<div>Frontiers: <span id='slam-frontiers'>0</span> frontiers</div>"
                    "</div>"
                    "</div>"
                    "<div class='map-visualization'>"
                    "<h4>Live Map</h4>"
                    "<canvas id='slam-map' width='400' height='400' style='border:1px solid #ddd;background:#000;display:block;width:100%;max-width:400px;height:auto;margin:0 auto'></canvas>"
                    "<div style='text-align:center;margin-top:0.5rem;font-size:0.8em'>"
                    "<span style='color:#0000ff'>● Robot</span> "
                    "<span style='color:#ff0000'>● Obstacles</span> "
                    "<span style='color:#00ff00'>● Path</span> "
                    "<span style='color:#ff00ff'>● Goal</span> "
                    "<span style='color:#ffff00'>● Features</span>"
                    "</div>"
                    "</div>"
                    "<div class='slam-controls' style='margin-top:1rem'>"
                    "<h4>SLAM Controls</h4>"
                    "<div style='display:flex;gap:0.5rem;align-items:center;margin-bottom:0.5rem'>"
                    "<input type='number' id='goal-x' placeholder='X (m)' step='0.1' style='width:70px;padding:0.3rem'>"
                    "<input type='number' id='goal-y' placeholder='Y (m)' step='0.1' style='width:70px;padding:0.3rem'>"
                    "<button onclick='slam.setGoal()' style='padding:0.3rem 0.6rem'>Set Goal</button>"
                    "</div>"
                    "<div style='display:flex;gap:0.5rem;flex-wrap:wrap'>"
                    "<button onclick='slam.startExploration()' style='font-size:0.9em;padding:0.3rem 0.6rem'>🔍 Explore</button>"
                    "<button onclick='slam.returnHome()' style='font-size:0.9em;padding:0.3rem 0.6rem'>🏠 Return Home</button>"
                    "<button onclick='slam.exportMap()' style='font-size:0.9em;padding:0.3rem 0.6rem'>💾 Export Map</button>"
                    "</div>"
                    "</div>"
                    "</div>"
                    
                    "<button class='collapsible' onclick='toggleSection(this)'>AI Perception</button>"
                    "<div class='content'>"
                    "<div id='perception-summary' style='margin-bottom:1rem'></div>"
                    "<div id='perception-details' style='font-size:0.85em;color:#555'></div>"
                    "</div>"
                    
                    "<button class='collapsible' onclick='toggleSection(this)'>Learning Statistics</button>"
                    "<div class='content'>"
                    "<div id='learning-stats'></div>"
                    "</div>"
                    
                    "<button class='collapsible' onclick='toggleSection(this)'>Navigation Map</button>"
                    "<div class='content'>"
                    "<img id='map' style='width:100%;max-width:500px;border-radius:4px'>"
                    "</div>"
                    
                    "<button class='collapsible' onclick='toggleSection(this)'>System Details</button>"
                    "<div class='content'>"
                    "<pre id='raw-status' style='font-size:0.7em;background:#f8f9fa;padding:1rem;border-radius:4px;overflow:auto;max-height:300px'></pre>"
                    "</div>"
                    "<script>const proto=location.protocol;const host=location.hostname;const camBase=proto+'//'+host+':9000';"
                    "function loadStream(){const img=document.getElementById('stream');img.style.border='3px solid orange';img.alt='Loading...';img.src=camBase+'/mjpg?t='+Date.now();}"
                    "setTimeout(loadStream,2000);"  # Delay initial load to ensure Vilib is ready
                    "document.getElementById('openCam').onclick=()=>window.open(camBase,'_blank');"
                    "document.getElementById('refreshCam').onclick=loadStream;"
                    
                    # Collapsible section handler
                    "function toggleSection(btn){const content=btn.nextElementSibling;if(content.style.display==='block'){content.style.display='none'}else{content.style.display='block'}}"
                    
                    # Enhanced formatting functions
                    "function formatPerceptionSummary(p){if(!p)return 'No perception data';"
                    "return `Scene: ${p.scene_description} | Safety: ${p.safety_assessment} | FPS: ${p.fps?.toFixed(1) || 0}`;}"
                    
                    "function formatPerceptionDetails(p){if(!p)return '';"
                    "let html='<strong>Performance:</strong> '+p.fps.toFixed(1)+' FPS, '+p.inference_ms.toFixed(1)+'ms latency<br>';"
                    "if(p.recommended_action)html+='<strong>Recommended Action:</strong> '+p.recommended_action+'<br>';"
                    "if(p.dominant_objects && p.dominant_objects.length)html+='<strong>Detected Objects:</strong> '+p.dominant_objects.join(', ')+'<br>';"
                    "if(p.objects && p.objects.length){html+='<strong>Object Details:</strong><br>';p.objects.slice(0,3).forEach(o=>{html+=`• ${o.label} (${o.confidence_level}, ${o.threat_level})<br>`;});}"
                    "return html;}"
                    
                    "function formatLearningStats(l){if(!l)return 'No learning data';"
                    "let html=`<div style='display:grid;grid-template-columns:1fr 1fr;gap:1rem'>`;"
                    "html+=`<div><strong>Actions:</strong> ${l.total_actions}</div><div><strong>Success Rate:</strong> ${(l.success_rate*100).toFixed(1)}%</div>`;"
                    "html+=`<div><strong>Exploration:</strong> ${(l.current_exploration*100).toFixed(1)}%</div><div><strong>Adaptations:</strong> ${l.adaptation_triggers}</div>`;"
                    "html+=`<div><strong>Best Speed:</strong> ${l.best_arm || 'Learning...'}</div><div><strong>Runtime:</strong> ${l.session_time_minutes?.toFixed(1) || 0}min</div>`;"
                    "html+=`</div>`;"
                    "if(l.arm_values){html+='<details style=\"margin-top:1rem\"><summary>Speed Arm Performance</summary><div style=\"font-size:0.8em;margin-top:0.5rem\">';Object.entries(l.arm_values).forEach(([arm,val])=>{html+=`${arm}: ${val.toFixed(3)} (${l.arm_counts[arm]} tries)<br>`;});html+='</div></details>';}"
                    "return html;}"
                    
                    # Main refresh function
                    "async function refresh(){"
                    "try{"
                    "const r=await fetch('/status');const j=await r.json();"
                    
                    # Update main metrics
                    "document.getElementById('mode-value').textContent=j.mode || '--';"
                    "document.getElementById('distance-value').textContent=j.distance_cm ? j.distance_cm.toFixed(1) : '--';"
                    "document.getElementById('battery-value').textContent=j.battery ? j.battery.voltage_v.toFixed(1) : '--';"
                    "const safetyEl=document.getElementById('safety-value');const safetyMetric=document.getElementById('safety-metric');"
                    "if(j.perception && j.perception.safety_assessment){"
                    "safetyEl.textContent=j.perception.safety_assessment;"
                    "safetyMetric.className='metric '+j.perception.safety_assessment;"
                    "}else{safetyEl.textContent='Unknown';safetyMetric.className='metric';}"
                    
                    # Update perception sections
                    "document.getElementById('perception-summary').innerHTML=formatPerceptionSummary(j.perception);"
                    "document.getElementById('perception-details').innerHTML=formatPerceptionDetails(j.perception);"
                    
                    # Update learning stats
                    "document.getElementById('learning-stats').innerHTML=formatLearningStats(j.learning);"
                    
                    # Update raw status (collapsed by default)
                    "document.getElementById('raw-status').textContent=JSON.stringify(j,null,2);"
                    
                    # Update map
                    "const m=await fetch('/map');if(m.ok){const mb=await m.blob();document.getElementById('map').src=URL.createObjectURL(mb);}"
                    
                    # Update SLAM data
                    "if(j.slam){updateSlamDisplay(j.slam);}"
                    
                    "}catch(e){console.error('Refresh error:',e);}"
                    "}"
                    
                    # SLAM visualization and control functions
                    "const slam={"
                    "setHome:async()=>{try{await fetch('/slam/set_home',{method:'POST'});console.log('Home position set');}catch(e){console.error('Set home error:',e);}},"
                    "clearMap:async()=>{if(confirm('Clear entire map?')){try{await fetch('/slam/clear',{method:'POST'});console.log('Map cleared');}catch(e){console.error('Clear map error:',e);}}},"
                    "setGoal:async()=>{const x=parseFloat(document.getElementById('goal-x').value);const y=parseFloat(document.getElementById('goal-y').value);if(!isNaN(x)&&!isNaN(y)){try{await fetch(`/slam/set_goal?x=${x}&y=${y}`,{method:'POST'});console.log(`Goal set to (${x}, ${y})`);}catch(e){console.error('Set goal error:',e);}}},"
                    "startExploration:async()=>{try{await fetch('/slam/explore',{method:'POST'});console.log('Exploration started');}catch(e){console.error('Start exploration error:',e);}},"
                    "returnHome:async()=>{try{await fetch('/slam/return_home',{method:'POST'});console.log('Returning home');}catch(e){console.error('Return home error:',e);}},"
                    "exportMap:async()=>{try{const r=await fetch('/slam/export');const blob=await r.blob();const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='slam_map.json';a.click();}catch(e){console.error('Export map error:',e);}}"
                    "};"
                    
                    "function updateSlamDisplay(slamData){"
                    "if(!slamData)return;"
                    "document.getElementById('slam-mode').textContent=slamData.navigation?.mode||'unknown';"
                    "document.getElementById('slam-x').textContent=slamData.pose?.x?.toFixed(2)||'0.0';"
                    "document.getElementById('slam-y').textContent=slamData.pose?.y?.toFixed(2)||'0.0';"
                    "document.getElementById('slam-confidence').textContent=slamData.pose?.confidence?(slamData.pose.confidence*100).toFixed(0)+'%':'0%';"
                    "document.getElementById('slam-area').textContent=slamData.statistics?.area_explored?.toFixed(2)||'0.0';"
                    "document.getElementById('slam-distance').textContent=slamData.statistics?.distance_traveled?.toFixed(1)||'0.0';"
                    "document.getElementById('slam-features').textContent=slamData.statistics?.features||'0';"
                    "document.getElementById('slam-obstacles').textContent=slamData.statistics?.obstacles||'0';"
                    "const goalText=slamData.navigation?.current_goal?`(${slamData.navigation.current_goal[0].toFixed(1)}, ${slamData.navigation.current_goal[1].toFixed(1)})`:'None';"
                    "document.getElementById('slam-goal').textContent=goalText;"
                    "const homeText=slamData.navigation?.home_position?`(${slamData.navigation.home_position[0].toFixed(1)}, ${slamData.navigation.home_position[1].toFixed(1)})`:'None';"
                    "document.getElementById('slam-home').textContent=homeText;"
                    "document.getElementById('slam-path').textContent=slamData.navigation?.path_length||'0';"
                    "document.getElementById('slam-frontiers').textContent=slamData.navigation?.frontiers||'0';"
                    "drawSlamMap(slamData);"
                    "}"
                    
                    "function drawSlamMap(slamData){"
                    "const canvas=document.getElementById('slam-map');if(!canvas||!slamData)return;"
                    "const ctx=canvas.getContext('2d');ctx.fillStyle='#000';ctx.fillRect(0,0,canvas.width,canvas.height);"
                    "const mapConfig=slamData.map_config;if(!mapConfig)return;"
                    "const scaleX=canvas.width/mapConfig.width;const scaleY=canvas.height/mapConfig.height;"
                    
                    # Draw occupancy map
                    "if(slamData.occupancy_map){"
                    "const occMap=slamData.occupancy_map;const explMap=slamData.exploration_map||[];"
                    "for(let y=0;y<occMap.length;y++){"
                    "for(let x=0;x<occMap[y].length;x++){"
                    "const px=Math.floor(x*scaleX);const py=Math.floor(y*scaleY);"
                    "const occ=occMap[y][x];const expl=explMap[y]?explMap[y][x]:0;"
                    "if(expl===4){ctx.fillStyle='#ff0000';}else if(expl===3){const intensity=Math.floor((1-occ)*255);ctx.fillStyle=`rgb(${intensity},${intensity},${intensity})`;}else{ctx.fillStyle='#404040';}"
                    "ctx.fillRect(px,py,Math.max(1,Math.ceil(scaleX)),Math.max(1,Math.ceil(scaleY)));"
                    "}}}"
                    
                    # Draw path
                    "if(slamData.path&&slamData.path.length>1){"
                    "ctx.strokeStyle='#00ff00';ctx.lineWidth=2;ctx.beginPath();"
                    "slamData.path.forEach((point,i)=>{const px=(point[0]-mapConfig.origin[0])*scaleX;const py=(point[1]-mapConfig.origin[1])*scaleY;if(i===0){ctx.moveTo(px,py);}else{ctx.lineTo(px,py);}});ctx.stroke();}"
                    
                    # Draw robot
                    "if(slamData.pose){"
                    "const robotX=(slamData.pose.x/mapConfig.resolution+mapConfig.origin[0])*scaleX;"
                    "const robotY=(slamData.pose.y/mapConfig.resolution+mapConfig.origin[1])*scaleY;"
                    "ctx.fillStyle='#0000ff';ctx.beginPath();ctx.arc(robotX,robotY,5,0,2*Math.PI);ctx.fill();"
                    "const arrowLen=15;const endX=robotX+arrowLen*Math.cos(slamData.pose.yaw||0);"
                    "const endY=robotY+arrowLen*Math.sin(slamData.pose.yaw||0);"
                    "ctx.strokeStyle='#0000ff';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(robotX,robotY);ctx.lineTo(endX,endY);ctx.stroke();"
                    # Draw arrow head
                    "const headLen=5;const headAngle=Math.PI/6;"
                    "ctx.beginPath();ctx.moveTo(endX,endY);"
                    "ctx.lineTo(endX-headLen*Math.cos(slamData.pose.yaw-headAngle),endY-headLen*Math.sin(slamData.pose.yaw-headAngle));"
                    "ctx.moveTo(endX,endY);"
                    "ctx.lineTo(endX-headLen*Math.cos(slamData.pose.yaw+headAngle),endY-headLen*Math.sin(slamData.pose.yaw+headAngle));"
                    "ctx.stroke();}"
                    
                    # Draw goal
                    "if(slamData.navigation?.current_goal){"
                    "const goalX=(slamData.navigation.current_goal[0]/mapConfig.resolution+mapConfig.origin[0])*scaleX;"
                    "const goalY=(slamData.navigation.current_goal[1]/mapConfig.resolution+mapConfig.origin[1])*scaleY;"
                    "ctx.strokeStyle='#ff00ff';ctx.lineWidth=3;ctx.beginPath();ctx.arc(goalX,goalY,8,0,2*Math.PI);ctx.stroke();}"
                    
                    # Draw features
                    "if(slamData.features){"
                    "Object.values(slamData.features).forEach(feature=>{"
                    "const featX=(feature.position[0]/mapConfig.resolution+mapConfig.origin[0])*scaleX;"
                    "const featY=(feature.position[1]/mapConfig.resolution+mapConfig.origin[1])*scaleY;"
                    "ctx.fillStyle='#ffff00';ctx.beginPath();ctx.arc(featX,featY,3,0,2*Math.PI);ctx.fill();});}"
                    "}"
                    
                    "setInterval(refresh,2000);refresh();"  # Slower refresh rate to reduce data flooding
                    "</script>"
                    "</body></html>"
                ).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type','text/html')
                self.send_header('Content-Length', str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                self.wfile.write(html)

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/index"):
                    return self._serve_index()
                if self.path.startswith("/status"):
                    try:
                        data = get_status()
                    except Exception as e:
                        data = {"error": str(e)}
                    body = json.dumps(data).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path.startswith("/heartbeat"):
                    if on_beat:
                        try:
                            on_beat()
                        except Exception:
                            pass
                    self.send_response(204)
                    self.end_headers()
                elif self.path.startswith("/mode"):
                    qs = parse_qs(urlparse(self.path).query)
                    m = (qs.get('m') or [''])[0]
                    if on_mode and m in ("auto","manual","idle"):
                        try:
                            on_mode(m)
                        except Exception:
                            pass
                    self.send_response(204)
                    self.end_headers()
                elif self.path.startswith("/manual"):
                    qs = parse_qs(urlparse(self.path).query)
                    sp = qs.get('speed'); st = qs.get('steer')
                    spd = qs.get('speed_delta'); std = qs.get('steer_delta')
                    try:
                        speed = int(sp[0]) if sp else None
                        steer = int(st[0]) if st else None
                        speed_delta = int(spd[0]) if spd else None
                        steer_delta = int(std[0]) if std else None
                    except Exception:
                        speed = steer = None
                        speed_delta = steer_delta = None
                    if on_manual and (speed is not None or steer is not None or speed_delta is not None or steer_delta is not None):
                        try:
                            # Prefer 4-arg handler: (speed, steer, speed_delta, steer_delta)
                            on_manual(speed, steer, speed_delta, steer_delta)
                        except TypeError:
                            # Fallback to legacy 2-arg handler: (speed, steer)
                            try:
                                on_manual(speed or 0, steer or 0)
                            except Exception:
                                pass
                        except Exception:
                            pass
                    self.send_response(204)
                    self.end_headers()
                elif self.path.startswith("/camera"):
                    qs = parse_qs(urlparse(self.path).query)
                    pn = qs.get('pan'); tl = qs.get('tilt')
                    pnd = qs.get('pan_delta'); tld = qs.get('tilt_delta')
                    center = qs.get('center')
                    try:
                        pan = int(pn[0]) if pn else None
                        tilt = int(tl[0]) if tl else None
                        pan_delta = int(pnd[0]) if pnd else None
                        tilt_delta = int(tld[0]) if tld else None
                        do_center = True if center and center[0] not in ("0","false","False") else False
                    except Exception:
                        pan = tilt = pan_delta = tilt_delta = None
                        do_center = False
                    if on_cam and (pan is not None or tilt is not None or pan_delta is not None or tilt_delta is not None or do_center):
                        try:
                            on_cam(pan, tilt, pan_delta, tilt_delta, do_center)
                        except Exception:
                            pass
                    self.send_response(204)
                    self.end_headers()
                elif self.path.startswith("/snapshot"):
                    # Redirect to Vilib snapshot (if needed)
                    self.send_response(204)
                    self.end_headers()
                elif self.path.startswith("/stream"):
                    # Redirect to Vilib stream using the same host
                    host = self.headers.get('Host') or f"{self.server.server_address[0]}:{self.server.server_address[1]}"
                    # Extract just the hostname part in case a port is present
                    try:
                        hostname = host.split(':')[0]
                    except Exception:
                        hostname = host
                    url = f"http://{hostname}:9000/mjpg"
                    self.send_response(302)
                    self.send_header('Location', url)
                    self.end_headers()
                elif self.path.startswith("/map"):
                    data = get_map() if get_map else None
                    if data:
                        self.send_response(200)
                        self.send_header('Content-Type','image/jpeg')
                        self.send_header('Content-Length', str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                    else:
                        self.send_response(204)
                        self.end_headers()
                elif self.path.startswith("/slam/"):
                    # Enhanced SLAM endpoints
                    self._handle_slam_request()
                else:
                    self.send_response(404)
                    self.end_headers()

            def _handle_slam_request(self):
                """Handle SLAM-related requests"""
                try:
                    if self.path == "/slam/set_home":
                        if self.server._on_slam:
                            result = self.server._on_slam("set_home", {})
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(b'{"status":"ok"}')
                    
                    elif self.path == "/slam/clear":
                        if self.server._on_slam:
                            result = self.server._on_slam("clear", {})
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(b'{"status":"ok"}')
                    
                    elif self.path.startswith("/slam/set_goal"):
                        qs = parse_qs(urlparse(self.path).query)
                        try:
                            x = float((qs.get('x') or ['0'])[0])
                            y = float((qs.get('y') or ['0'])[0])
                            if self.server._on_slam:
                                result = self.server._on_slam("set_goal", {"x": x, "y": y})
                            self.send_response(200)
                            self.send_header('Content-Type', 'application/json')
                            self.end_headers()
                            self.wfile.write(f'{{"status":"ok","goal":[{x},{y}]}}'.encode())
                        except ValueError:
                            self.send_response(400)
                            self.send_header('Content-Type', 'application/json')
                            self.end_headers()
                            self.wfile.write(b'{"status":"error","message":"Invalid coordinates"}')
                    
                    elif self.path == "/slam/explore":
                        if self.server._on_slam:
                            result = self.server._on_slam("explore", {})
                            status = "ok" if result else "no_targets"
                        else:
                            status = "unavailable"
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(f'{{"status":"{status}"}}'.encode())
                    
                    elif self.path == "/slam/return_home":
                        if self.server._on_slam:
                            result = self.server._on_slam("return_home", {})
                            status = "ok" if result else "no_home"
                        else:
                            status = "unavailable"
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(f'{{"status":"{status}"}}'.encode())
                    
                    elif self.path == "/slam/export":
                        if self.server._on_slam:
                            map_data = self.server._on_slam("export", {})
                            if map_data:
                                json_data = json.dumps(map_data, indent=2).encode()
                                self.send_response(200)
                                self.send_header('Content-Type', 'application/json')
                                self.send_header('Content-Disposition', 'attachment; filename="slam_map.json"')
                                self.send_header('Content-Length', str(len(json_data)))
                                self.end_headers()
                                self.wfile.write(json_data)
                            else:
                                self.send_response(404)
                                self.end_headers()
                        else:
                            self.send_response(404)
                            self.end_headers()
                    
                    elif self.path == "/slam/status":
                        if self.server._on_slam:
                            status_data = self.server._on_slam("status", {})
                            if status_data:
                                json_data = json.dumps(status_data).encode()
                                self.send_response(200)
                                self.send_header('Content-Type', 'application/json')
                                self.send_header('Content-Length', str(len(json_data)))
                                self.end_headers()
                                self.wfile.write(json_data)
                            else:
                                self.send_response(404)
                                self.end_headers()
                        else:
                            self.send_response(404)
                            self.end_headers()
                    
                    else:
                        self.send_response(404)
                        self.end_headers()
                        
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(f'{{"status":"error","message":"{str(e)}"}}'.encode())

            def log_message(self, format, *args):
                log.debug("HTTP: " + format % args)

        self._srv = _ThreadingHTTPServer((self._addr, self._port), Handler)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        self.log.info(f"Telemetry server started on http://{self._addr}:{self._port}")

    def stop(self):
        if self._srv:
            self._srv.shutdown()
            self._srv.server_close()
            self._srv = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self.log.info("Telemetry server stopped")
