import * as THREE from 'three';
import {ColonyScene} from '../scene';
import type {Tape} from './model';

const emptyField=btoa('\0'.repeat(96*64*2));
export class WorldView {
  readonly scene:ColonyScene;
  private source:THREE.Mesh;
  private nestRange:THREE.Mesh;
  private deaths:THREE.Mesh[]=[];
  private tape:Tape|null=null;
  private version=0;
  private lastTick=-1;
  constructor(host:HTMLElement,onSelect:(id:number)=>void){
    this.scene=new ColonyScene(host,true);this.scene.showField=false;
    this.scene.onPoint=(_x,_y,id)=>{if(id!==null)onSelect(id);};
    const nest=new THREE.Mesh(new THREE.CylinderGeometry(.65,.65,.06,48),new THREE.MeshStandardMaterial({color:0x368777}));
    nest.position.y=.05;this.scene.scene.add(nest);
    this.nestRange=new THREE.Mesh(new THREE.RingGeometry(.99,1,64),new THREE.MeshBasicMaterial({color:0x50c5de,side:THREE.DoubleSide,transparent:true,opacity:.65}));
    this.nestRange.rotation.x=-Math.PI/2;this.nestRange.position.y=.025;this.scene.scene.add(this.nestRange);
    this.source=new THREE.Mesh(new THREE.RingGeometry(.96,1,48),new THREE.MeshBasicMaterial({color:0xe96c6c,side:THREE.DoubleSide,transparent:true,opacity:.9}));
    this.source.rotation.x=-Math.PI/2;this.scene.scene.add(this.source);this.source.visible=false;
    for(let i=0;i<8;i++){
      const mark=new THREE.Mesh(new THREE.RingGeometry(.22,.27,20),new THREE.MeshBasicMaterial({color:0xf47272,side:THREE.DoubleSide}));
      mark.rotation.x=-Math.PI/2;mark.visible=false;this.scene.scene.add(mark);this.deaths.push(mark);
    }
  }
  setTape(tape:Tape):void{this.tape=tape;this.version++;this.lastTick=-1;this.nestRange.scale.setScalar(tape.header.nest_radius);this.focus();}
  focus():void{
    if(!this.tape)return;const [x,z]=this.tape.header.food;
    this.scene.controls.target.set(x/2,0,z/2);this.scene.camera.position.set(x/2+1,7,z/2+5);
    this.scene.camera.lookAt(x/2,0,z/2);this.scene.controls.update();
  }
  render(tick:number,individual:number):void{
    const tape=this.tape;if(!tape)return;
    const actual=Math.min(tick,tape.frames.length),frame=actual?tape.frames[actual-1]:null;
    if(this.lastTick>=0&&(actual<this.lastTick||actual-this.lastTick>1))this.version++;
    this.lastTick=actual;
    const count=tape.counts[actual];
    this.scene.selected=individual;
    this.scene.update({mode:tape.header.result.arm==='rules'?'rules':'neural',tick:actual,seconds:actual*.1,seed:this.version,
      delivered:0,field_width:96,field_height:64,pheromones:emptyField,walls:[],
      foods:[{id:0,x:tape.header.food[0],y:tape.header.food[1],amount:48-count.pickups}],
      ants:tape.header.initial_positions.map((position,id)=>{
        const ant=frame?.ants[id],p=ant?.position??position,heading=ant?.heading??tape.header.initial_headings[id];
        return {id,x:p[0],y:p[1],heading,carrying:ant?.carrying??false,
          frozen:!!ant?.exhausted||!['learned','always'].includes(tape.header.result.arm),rays:[],sense_x:p[0],sense_y:p[1],sense_heading:heading};
      })});
    this.source.visible=!!frame?.source_active&&tape.header.source_strength>0;
    if(frame){this.source.position.set(frame.source_position[0],.04,frame.source_position[1]);
      const radius=tape.header.result.condition==='moving-danger'?tape.header.contact_radius:.9;this.source.scale.setScalar(radius);}
    this.deaths.forEach((mark,id)=>{const ant=frame?.ants[id];mark.visible=!!ant?.exhausted;if(ant)mark.position.set(ant.position[0],.05,ant.position[1]);});
  }
}
