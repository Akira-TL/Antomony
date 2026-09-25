import * as THREE from 'three';

interface Placement {valid:boolean;message:string}

/** 资源预览只查询同一份放置规则，不向学习器发布食物坐标。 */
export class FoodPreview {
  onStatus:(message:string)=>void=()=>{};
  private revision=0;
  private timer=0;
  private checkedAt=0;
  private point:THREE.Vector2|null=null;
  private material=new THREE.MeshBasicMaterial({color:0xffd18b,transparent:true,opacity:.35,depthWrite:false});
  private mesh=new THREE.Mesh(new THREE.CylinderGeometry(.85,.85,.6,24),this.material);
  constructor(scene:THREE.Scene,private host:HTMLElement){this.mesh.visible=false;scene.add(this.mesh);}
  move(x:number,y:number):void {
    this.point=new THREE.Vector2(x,y);this.revision++;
    this.mesh.visible=true;this.mesh.position.set(x,.3,y);
    this.material.color.setHex(0xffd18b);this.host.dataset.foodPreview='pending';
    if(!this.timer)this.timer=window.setTimeout(()=>{this.timer=0;void this.check();},100);
  }
  refresh():void {
    if(this.point&&!this.timer&&performance.now()-this.checkedAt>600){this.timer=window.setTimeout(()=>{this.timer=0;void this.check();},100);}
  }
  hide():void {this.revision++;this.point=null;clearTimeout(this.timer);this.timer=0;this.mesh.visible=false;delete this.host.dataset.foodPreview;this.onStatus('');}
  private async check():Promise<void>{
    const p=this.point;if(!p)return;const revision=this.revision;this.checkedAt=performance.now();
    try{
      const response=await fetch(`/api/food-preview?x=${p.x}&y=${p.y}`,{signal:AbortSignal.timeout(2000)});
      if(revision!==this.revision)return;
      if(!response.ok)throw new Error('invalid placement');
      const result=await response.json() as Placement;
      if(revision!==this.revision)return;
      this.material.color.setHex(result.valid?0xffd18b:0xff7676);
      this.host.dataset.foodPreview=result.valid?'valid':'invalid';this.onStatus(result.message+' · 点击放置');
    }catch{if(revision===this.revision){this.host.dataset.foodPreview='invalid';this.material.color.setHex(0xff7676);this.onStatus('此位置暂不可放置');}}
  }
}
