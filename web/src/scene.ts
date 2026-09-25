import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {Frame,Tool} from './types';

export class ColonyScene {
  readonly renderer:THREE.WebGLRenderer;
  readonly scene=new THREE.Scene();
  readonly camera=new THREE.PerspectiveCamera(43,1,.1,180);
  readonly controls:OrbitControls;
  tool:Tool='inspect'; selected=0; showField=true; fps=0;
  onPoint:(x:number,y:number,id:number|null)=>void=()=>{};
  private frame:Frame|null=null;
  private previous:Frame|null=null;
  private received=0;
  private antMeshes:THREE.InstancedMesh[]=[];
  private legs:THREE.InstancedMesh;
  private cargo:THREE.InstancedMesh;
  private delivered:THREE.InstancedMesh;
  private wallGroup=new THREE.Group();
  private foodGroup=new THREE.Group();
  private wallKey=''; private foodKey='';
  private fieldTexture:THREE.DataTexture;
  private fieldMesh:THREE.Mesh;
  private ring:THREE.Mesh;
  private rays=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0xb3f6cd,transparent:true,opacity:.35}));
  private pointer=new THREE.Vector2(); private raycaster=new THREE.Raycaster();
  private dummy=new THREE.Object3D();
  private last=performance.now();private frameCounter=0;private fpsStart=performance.now();

  constructor(private host:HTMLElement) {
    this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.8));
    this.renderer.shadowMap.enabled=true;
    this.renderer.shadowMap.type=THREE.PCFSoftShadowMap;
    this.renderer.setClearColor(0x091214,1);
    this.renderer.toneMapping=THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure=1.35;
    host.append(this.renderer.domElement);
    this.scene.fog=new THREE.Fog(0x091214,50,105);
    this.camera.position.set(24,29,29);this.camera.lookAt(0,0,0);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);
    this.controls.enableDamping=true;this.controls.dampingFactor=.08;
    this.controls.maxPolarAngle=Math.PI*.46;this.controls.minDistance=17;this.controls.maxDistance=65;
    const ambient=new THREE.HemisphereLight(0xc4efd5,0x172932,2.1);this.scene.add(ambient);
    const sun=new THREE.DirectionalLight(0xfaf4dc,3);sun.position.set(-9,24,12);sun.castShadow=true;
    sun.shadow.mapSize.set(2048,2048);Object.assign(sun.shadow.camera,{left:-20,right:20,top:16,bottom:-16,near:1,far:65});sun.shadow.bias=-.0004;
    this.scene.add(sun,new THREE.PointLight(0x85e8b1,35,45));
    const base=new THREE.Mesh(new THREE.BoxGeometry(28.8,.65,20.8),new THREE.MeshStandardMaterial({color:0x142c2b,roughness:.8,metalness:.1}));base.position.y=-.38;base.receiveShadow=true;this.scene.add(base);
    const floor=new THREE.Mesh(new THREE.PlaneGeometry(28,20),new THREE.MeshStandardMaterial({color:0x17342e,roughness:.92}));floor.rotation.x=-Math.PI/2;floor.receiveShadow=true;this.scene.add(floor);
    const grid=new THREE.GridHelper(28,56,0x416454,0x28483e);grid.scale.z=20/28;grid.position.y=.009;(grid.material as THREE.Material).transparent=true;(grid.material as THREE.Material).opacity=.28;this.scene.add(grid);
    const border=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(28.5,.06,20.5)),new THREE.LineBasicMaterial({color:0x82b49a,transparent:true,opacity:.5}));border.position.y=.02;this.scene.add(border);
    const nest=new THREE.Mesh(new THREE.CylinderGeometry(1.65,1.8,.18,64),new THREE.MeshStandardMaterial({color:0x356b54,metalness:.45,roughness:.36}));nest.position.set(-10,.08,0);this.scene.add(nest);
    for(const r of [1.25,1.85]){const torus=new THREE.Mesh(new THREE.TorusGeometry(r,.018,8,90),new THREE.MeshBasicMaterial({color:0xa8e6b5}));torus.rotation.x=Math.PI/2;torus.position.set(-10,.19,0);this.scene.add(torus);}
    this.scene.add(this.label('巢穴 / HOME',-10,1.1,0,0xc5f7d4));
    const material=new THREE.MeshStandardMaterial({color:0xbcdbbf,metalness:.45,roughness:.38,emissive:0x3c7255,emissiveIntensity:.3});
    for(let part=0;part<3;part++){
      const mesh=new THREE.InstancedMesh(new THREE.SphereGeometry(1,12,8),material,64);mesh.castShadow=true;mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);this.antMeshes.push(mesh);this.scene.add(mesh);
    }
    this.legs=new THREE.InstancedMesh(new THREE.CylinderGeometry(.014,.014,1,5),material,64*6);this.scene.add(this.legs);
    this.cargo=new THREE.InstancedMesh(new THREE.BoxGeometry(.18,.18,.18),new THREE.MeshStandardMaterial({color:0xf8b85b,emissive:0xb46616,emissiveIntensity:.6,metalness:.3,roughness:.35}),64);this.scene.add(this.cargo);
    this.delivered=new THREE.InstancedMesh(new THREE.BoxGeometry(.12,.12,.12),this.cargo.material,384);this.scene.add(this.delivered);
    this.fieldTexture=new THREE.DataTexture(new Uint8Array(96*64*4),96,64,THREE.RGBAFormat);this.fieldTexture.flipY=true;this.fieldTexture.magFilter=THREE.LinearFilter;this.fieldTexture.minFilter=THREE.LinearFilter;
    this.fieldMesh=new THREE.Mesh(new THREE.PlaneGeometry(28,20),new THREE.MeshBasicMaterial({map:this.fieldTexture,transparent:true,depthWrite:false,opacity:.8,blending:THREE.AdditiveBlending}));this.fieldMesh.rotation.x=-Math.PI/2;this.fieldMesh.position.y=.022;this.scene.add(this.fieldMesh);
    this.ring=new THREE.Mesh(new THREE.RingGeometry(.32,.36,48),new THREE.MeshBasicMaterial({color:0xd3ffb1,side:THREE.DoubleSide,transparent:true,opacity:.9}));this.ring.rotation.x=-Math.PI/2;this.scene.add(this.ring,this.rays,this.wallGroup,this.foodGroup);
    const resize=()=>{const w=host.clientWidth,h=host.clientHeight;this.renderer.setSize(w,h);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();};new ResizeObserver(resize).observe(host);resize();
    let down=new THREE.Vector2();
    this.renderer.domElement.addEventListener('pointerdown',e=>{down.set(e.clientX,e.clientY);});
    this.renderer.domElement.addEventListener('pointerup',e=>{if(e.button!==0||Math.hypot(e.clientX-down.x,e.clientY-down.y)>7)return;this.click(e);});
    this.animate();
  }

  private label(text:string,x:number,y:number,z:number,color:number):THREE.Sprite {
    const canvas=document.createElement('canvas');canvas.width=512;canvas.height=96;const ctx=canvas.getContext('2d')!;
    ctx.fillStyle='#'+color.toString(16).padStart(6,'0');ctx.font='500 32px system-ui';ctx.textAlign='center';ctx.fillText(text,256,57);
    const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(canvas),transparent:true,depthTest:false}));sprite.scale.set(4.2,.8,1);sprite.position.set(x,y,z);return sprite;
  }

  update(frame:Frame):void {
    this.previous=this.frame?.seed===frame.seed && this.frame.ants.length===frame.ants.length?this.frame:null;
    this.frame=frame;this.received=performance.now();
    const raw=atob(frame.pheromones),n=frame.field_width*frame.field_height;
    const rgba=this.fieldTexture.image.data as Uint8Array;
    for(let i=0;i<n;i++){const home=raw.charCodeAt(i),food=raw.charCodeAt(i+n);rgba[4*i]=food;rgba[4*i+1]=Math.min(255,home*.75+food*.68);rgba[4*i+2]=home*.9;rgba[4*i+3]=Math.min(200,Math.max(home,food)*1.5);}
    this.fieldTexture.needsUpdate=true;
    const wk=JSON.stringify(frame.walls);
    if(wk!==this.wallKey){this.clear(this.wallGroup);this.wallKey=wk;
      for(const w of frame.walls){const g=new THREE.BoxGeometry(w.hx*2,1.15,w.hy*2);const m=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:0x304950,roughness:.5,metalness:.28}));m.position.set(w.x,.58,w.y);m.castShadow=true;m.receiveShadow=true;this.wallGroup.add(m);const edge=new THREE.LineSegments(new THREE.EdgesGeometry(g),new THREE.LineBasicMaterial({color:0x97b9ae,transparent:true,opacity:.62}));edge.position.copy(m.position);this.wallGroup.add(edge);}
    }
    const fk=JSON.stringify(frame.foods.map(f=>[f.id,f.x,f.y,Math.ceil(f.amount/8)]));
    if(fk!==this.foodKey){this.clear(this.foodGroup);this.foodKey=fk;
      for(const f of frame.foods){const base=new THREE.Mesh(new THREE.CylinderGeometry(.8,.86,.08,32),new THREE.MeshStandardMaterial({color:0x665038,roughness:.8}));base.position.set(f.x,.05,f.y);this.foodGroup.add(base);
        for(let i=0;i<Math.min(27,Math.ceil(f.amount/8));i++){const m=new THREE.Mesh(new THREE.BoxGeometry(.23,.23,.23),new THREE.MeshStandardMaterial({color:0xf8b95f,emissive:0x8d4e15,emissiveIntensity:.35,roughness:.38}));m.position.set(f.x+((i%3)-1)*.26,.2+Math.floor(i/9)*.25,f.y+(Math.floor(i/3)%3-1)*.26);m.castShadow=true;this.foodGroup.add(m);}this.foodGroup.add(this.label('资源 '+f.id,f.x,1.5,f.y,0xffce88));}
    }
  }

  private clear(group:THREE.Group):void {
    for(const obj of group.children){if(obj instanceof THREE.Mesh||obj instanceof THREE.LineSegments){obj.geometry.dispose();if(Array.isArray(obj.material))obj.material.forEach(m=>m.dispose());else obj.material.dispose();}if(obj instanceof THREE.Sprite){obj.material.map?.dispose();obj.material.dispose();}}
    group.clear();
  }

  private click(event:PointerEvent):void {
    const rect=this.renderer.domElement.getBoundingClientRect();this.pointer.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);
    this.raycaster.setFromCamera(this.pointer,this.camera);
    const p=new THREE.Vector3();this.raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0,1,0),0),p);
    if(Math.abs(p.x)>14||Math.abs(p.z)>10)return;
    let id:number|null=null;
    if(this.frame){const close=this.frame.ants.map(a=>({id:a.id,d:Math.hypot(a.x-p.x,a.y-p.z)})).sort((a,b)=>a.d-b.d)[0];if(close&&close.d<.85)id=close.id;}
    this.onPoint(p.x,p.z,id);
  }

  private animate=():void=>{
    requestAnimationFrame(this.animate);const now=performance.now();this.last=now;this.frameCounter++;
    if(now-this.fpsStart>1000){this.fps=this.frameCounter*1000/(now-this.fpsStart);this.frameCounter=0;this.fpsStart=now;}
    this.controls.enabled=this.tool==='inspect';this.controls.update();this.fieldMesh.visible=this.showField;
    const f=this.frame;
    if(f){const alpha=Math.min(1,(now-this.received)/110);let carrying=0;
      this.antMeshes.forEach(m=>m.count=f.ants.length);this.legs.count=f.ants.length*6;
      for(const ant of f.ants){const prev=this.previous?.ants[ant.id];const x=prev?THREE.MathUtils.lerp(prev.x,ant.x,alpha):ant.x;const z=prev?THREE.MathUtils.lerp(prev.y,ant.y,alpha):ant.y;
        const heading=ant.heading;const c=Math.cos(heading),s=Math.sin(heading);
        const sizes:[[number,number,number,number],[number,number,number,number],[number,number,number,number]]=[[-.075,.125,.075,.1],[.06,.08,.07,.075],[.15,.05,.055,.055]];
        sizes.forEach(([offset,sx,sy,sz],index)=>{this.dummy.position.set(x+c*offset,.13,z+s*offset);this.dummy.rotation.set(0,-heading,0);this.dummy.scale.set(sx,sy,sz);this.dummy.updateMatrix();this.antMeshes[index].setMatrixAt(ant.id,this.dummy.matrix);this.antMeshes[index].setColorAt(ant.id,new THREE.Color(ant.id===this.selected?0xdfffaf:ant.frozen?0x91a5b4:ant.carrying?0xffc579:0xb6d9c3));});
        for(let leg=0;leg<6;leg++){const side=leg<3?-1:1;const along=(leg%3-1)*.11;const gait=f.paused?0:Math.sin(f.seconds*17+leg*2+ant.id)*.025;const start=new THREE.Vector3(x+c*along,.14,z+s*along),end=new THREE.Vector3(x+c*(along+gait)-s*.24*side,.032,z+s*(along+gait)+c*.24*side);this.dummy.position.copy(start).add(end).multiplyScalar(.5);this.dummy.scale.set(1,start.distanceTo(end),1);this.dummy.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),end.clone().sub(start).normalize());this.dummy.updateMatrix();this.legs.setMatrixAt(ant.id*6+leg,this.dummy.matrix);}
        if(ant.carrying){this.dummy.position.set(x,.34,z);this.dummy.rotation.set(0,-heading,0);this.dummy.scale.set(1,1,1);this.dummy.updateMatrix();this.cargo.setMatrixAt(carrying++,this.dummy.matrix);}
        if(ant.id===this.selected){this.ring.position.set(x,.045,z);const points:number[]=[];ant.rays.forEach((d,i)=>{const angle=heading-Math.PI+i*Math.PI*2/12;points.push(x,.10,z,x+Math.cos(angle)*d,.10,z+Math.sin(angle)*d);});this.rays.geometry.dispose();this.rays.geometry=new THREE.BufferGeometry();this.rays.geometry.setAttribute('position',new THREE.Float32BufferAttribute(points,3));}
      }
      for(const m of this.antMeshes){m.instanceMatrix.needsUpdate=true;if(m.instanceColor)m.instanceColor.needsUpdate=true;}this.legs.instanceMatrix.needsUpdate=true;this.cargo.count=carrying;this.cargo.instanceMatrix.needsUpdate=true;
      this.delivered.count=Math.min(384,f.delivered);for(let i=0;i<this.delivered.count;i++){this.dummy.position.set(-10+(i%8-3.5)*.15,.26+Math.floor(i/64)*.13,(Math.floor(i/8)%8-3.5)*.15);this.dummy.rotation.set(0,0,0);this.dummy.scale.set(1,1,1);this.dummy.updateMatrix();this.delivered.setMatrixAt(i,this.dummy.matrix);}this.delivered.instanceMatrix.needsUpdate=true;
    }
    this.renderer.render(this.scene,this.camera);
  };
}
