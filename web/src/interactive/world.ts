import * as THREE from 'three';
import {ColonyScene} from '../scene';
import type {Edit,Frame,World} from './types';

const ARENA={x:17,y:12};
const HOME_RADIUS=2.5;

function dispose(group:THREE.Group):void{
  group.traverse(o=>{if(o instanceof THREE.Mesh||o instanceof THREE.LineSegments){o.geometry.dispose();const ms=Array.isArray(o.material)?o.material:[o.material];ms.forEach(m=>m.dispose());}});group.clear();
}
function ring(radius:number,color:number,opacity=1):THREE.Mesh{
  const m=new THREE.Mesh(new THREE.RingGeometry(Math.max(0,radius-.035),radius,64),new THREE.MeshBasicMaterial({color,transparent:true,opacity,side:THREE.DoubleSide,depthWrite:false}));
  m.rotation.x=-Math.PI/2;m.position.y=.04;return m;
}

export class LiveWorld {
  readonly scene:ColonyScene;
  private content=new THREE.Group();private preview=new THREE.Group();
  private foodKey='';private foods=new THREE.Group();private previewKey='';
  private stateKey='';private fieldKey='';
  private ids:number[]=[];private run='';private lastTick=-1;private version=0;private lastRevivals=0;
  private following=false;
  onMove:(p:{x:number;y:number}|null)=>void=()=>{};
  onPlace:(p:{x:number;y:number})=>void=()=>{};
  onWheel:(direction:number,p:{x:number;y:number}|null)=>boolean=()=>false;
  editing=false;world:World|null=null;
  constructor(host:HTMLElement,onSelect:(id:number)=>void){
    this.scene=new ColonyScene(host,true,ARENA);
    this.scene.onPoint=(_x,_y,id)=>{if(id!==null&&this.ids[id]!==undefined)onSelect(this.ids[id]);};
    this.scene.scene.add(this.content,this.preview,this.foods);
    const nest=new THREE.Mesh(new THREE.CylinderGeometry(HOME_RADIUS-.08,HOME_RADIUS,.08,64),new THREE.MeshStandardMaterial({color:0x438c79}));
    nest.position.y=.06;this.scene.scene.add(nest,ring(HOME_RADIUS+.2,0x66bcac,.5),ring(HOME_RADIUS,0x9be2ca));
    const canvas=this.scene.renderer.domElement;
    const ground=(e:Pick<PointerEvent,'clientX'|'clientY'>):{x:number;y:number}|null=>{
      const box=canvas.getBoundingClientRect(),v=new THREE.Vector2((e.clientX-box.left)/box.width*2-1,1-(e.clientY-box.top)/box.height*2);
      const ray=new THREE.Raycaster();ray.setFromCamera(v,this.scene.camera);
      const p=ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0,1,0),0),new THREE.Vector3());
      return p&&Math.abs(p.x)<=ARENA.x&&Math.abs(p.z)<=ARENA.y?{x:p.x,y:p.z}:null;
    };
    canvas.addEventListener('pointermove',e=>{if(this.editing)this.onMove(ground(e));});
    canvas.addEventListener('pointerleave',()=>this.onMove(null));
    let down=[0,0];canvas.addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY];});
    canvas.addEventListener('pointerup',e=>{if(!this.editing||e.button!==0||Math.hypot(e.clientX-down[0],e.clientY-down[1])>7)return;const p=ground(e);if(p)this.onPlace(p);});
    canvas.addEventListener('wheel',e=>{if(this.editing&&this.onWheel(Math.sign(e.deltaY),ground(e)))e.preventDefault();},{passive:false});
    this.scene.fitArena();
  }
  setEditing(value:boolean):void{this.editing=value;this.scene.interactive=!value;this.scene.renderer.domElement.style.cursor=value?'crosshair':'grab';}
  fit():void{this.following=false;this.scene.fitArena();}
  focus(individual:number):void{
    const ant=this.world?.ants[individual];if(!ant)return;
    this.following=true;this.scene.controls.target.set(ant.x,0,ant.y);
    this.scene.camera.position.set(ant.x,4,ant.y+5);this.scene.controls.update();
  }
  render(frame:Frame,world:World,selected:number,showField:boolean):void{
    this.world=world;
    if(this.run!==frame.run_id)this.fit();
    const ids=world.ants.filter(a=>!a.pending).map(a=>a.id);
    if(this.run!==frame.run_id||ids.join(',')!==this.ids.join(',')||frame.tick<this.lastTick||frame.tick-this.lastTick>4||this.lastRevivals!==world.counts.revivals)this.version++;
    this.run=frame.run_id;this.ids=ids;this.lastTick=frame.tick;this.lastRevivals=world.counts.revivals;
    this.scene.showField=showField;this.scene.selected=ids.indexOf(selected);
    const stateKey=`${frame.run_id}:${frame.tick}:${selected}:${world.walls.length}:${world.foods.length}:${world.traps.length}`;
    if(stateKey===this.stateKey&&world.field===this.fieldKey)return;
    this.stateKey=stateKey;this.fieldKey=world.field;
    if(this.following&&selected>=0){const ant=world.ants[selected],target=new THREE.Vector3(ant.x,0,ant.y);
      this.scene.camera.position.add(target.clone().sub(this.scene.controls.target));this.scene.controls.target.copy(target);this.scene.controls.update();}
    this.scene.update({mode:world.key==='rules'?'rules':'neural',seed:this.version,tick:frame.tick,seconds:frame.tick*.1,delivered:0,
      field_width:96,field_height:64,pheromones:world.field,walls:world.walls,foods:[],
      ants:ids.map((id,index)=>{const a=world.ants[id];return{id:index,x:a.x,y:a.y,heading:a.heading,carrying:a.carrying,
        frozen:world.key!=='adaptive'||!frame.learning,rays:[],sense_x:a.x,sense_y:a.y,sense_heading:a.heading};})});
    const key=JSON.stringify(world.foods);
    if(key!==this.foodKey){
      this.foodKey=key;dispose(this.foods);
      for(const f of world.foods){
        const marker=ring(.4,f.stock?0xf5ca6a:0x787e85,f.stock?.9:.4);marker.position.set(f.x,.045,f.y);this.foods.add(marker);
        if(!f.stock)continue;
        const food=new THREE.Mesh(new THREE.DodecahedronGeometry(.22),new THREE.MeshStandardMaterial({color:0xfac761,roughness:.6,emissive:0x704819,emissiveIntensity:.25}));
        food.position.set(f.x,.25,f.y);food.castShadow=true;this.foods.add(food);
      }
    }
    dispose(this.content);
    for(const t of world.traps){
      const color=t.spec.injury>0?0xf47078:0x8cb9ef;
      const radius=ring(t.spec.radius,color,t.active?1:.3);radius.position.set(t.position[0],.055,t.position[1]);this.content.add(radius);
      const scent=ring(t.spec.radius+1,color,.18);scent.position.set(t.position[0],.04,t.position[1]);this.content.add(scent);
      const fill=new THREE.Mesh(new THREE.CircleGeometry(t.spec.radius,48),new THREE.MeshBasicMaterial({color,transparent:true,opacity:t.active?.16:.035,depthWrite:false,side:THREE.DoubleSide}));
      fill.rotation.x=-Math.PI/2;fill.position.set(t.position[0],.035,t.position[1]);this.content.add(fill);
    }
    for(const a of world.ants.filter(a=>a.pending)){const m=ring(.28,0xf47078);m.position.set(a.x,.06,a.y);this.content.add(m);}
  }
  showPreview(edit:Edit|null,valid:boolean|null):void{
    if(!edit){this.preview.visible=false;return;}
    const color=valid===null?0xe7ce7c:valid?0x8cddaa:0xf1747a;
    const key=JSON.stringify({...edit,x:0,y:0,color});
    if(key!==this.previewKey){
      this.previewKey=key;dispose(this.preview);
      if(edit.kind==='wall'){
        const g=new THREE.BoxGeometry((edit.hx??.25)*2,1.15,(edit.hy??1.5)*2);
        const m=new THREE.Mesh(g,new THREE.MeshBasicMaterial({color,transparent:true,opacity:.3,depthWrite:false}));m.position.y=.58;m.rotation.y=-(edit.angle??0);this.preview.add(m);
        const edge=new THREE.LineSegments(new THREE.EdgesGeometry(g),new THREE.LineBasicMaterial({color}));edge.position.copy(m.position);edge.rotation.copy(m.rotation);this.preview.add(edge);
      }else{
        this.preview.add(ring(edit.kind==='food'?.4:edit.radius??.5,color));
        if(edit.kind==='trap')this.preview.add(ring((edit.radius??.5)+1,color,.3));
      }
    }
    this.preview.position.set(edit.x??0,0,edit.y??0);this.preview.visible=true;
  }
  eraseAt(x:number,y:number):Edit|null{
    if(!this.world)return null;
    for(const w of this.world.walls){const dx=x-w.x,dy=y-w.y,c=Math.cos(w.angle),s=Math.sin(w.angle);
      if(Math.abs(c*dx+s*dy)<=w.hx+.15&&Math.abs(-s*dx+c*dy)<=w.hy+.15)return{kind:'erase-wall',identifier:w.id,x:w.x,y:w.y,radius:Math.hypot(w.hx,w.hy)};
    }
    for(const t of this.world.traps)if(Math.hypot(x-t.position[0],y-t.position[1])<=t.spec.radius+.2)return{kind:'erase-trap',identifier:t.spec.id,x:t.position[0],y:t.position[1],radius:t.spec.radius};
    return null;
  }
}
