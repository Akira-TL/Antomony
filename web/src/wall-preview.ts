import * as THREE from 'three';

interface Placement {valid:boolean; message:string; displaced:number}

/** 预览与落墙均使用服务端同一套空间校验；预览不改变世界。 */
export class WallPreview {
  hx=.4; hy=2; angle=0;
  onStatus:(message:string)=>void=()=>{};
  private point:THREE.Vector2|null=null;
  private timer=0;
  private revision=0;
  private checkedAt=0;
  private material=new THREE.MeshBasicMaterial({color:0xffcd77,transparent:true,opacity:.32,depthWrite:false,depthTest:false});
  private mesh=new THREE.Mesh(new THREE.BoxGeometry(1,1,1),this.material);
  private edgeMaterial=new THREE.LineBasicMaterial({color:0xffcd77,transparent:true,opacity:.95,depthTest:false});
  private edges=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1,1,1)),this.edgeMaterial);

  constructor(scene:THREE.Scene,private host:HTMLElement) {
    this.mesh.renderOrder=10;this.edges.renderOrder=11;
    this.mesh.visible=this.edges.visible=false;scene.add(this.mesh,this.edges);
  }

  setSize(hx:number,hy:number):void {this.hx=hx;this.hy=hy;if(this.point)this.move(this.point.x,this.point.y);}

  rotate(delta:number):void {
    this.angle=Math.atan2(Math.sin(this.angle+delta),Math.cos(this.angle+delta));
    if(this.point)this.move(this.point.x,this.point.y);
  }

  move(x:number,y:number):void {
    this.point=new THREE.Vector2(x,y);this.revision++;
    for(const object of [this.mesh,this.edges]){object.visible=true;object.position.set(x,.6,y);object.scale.set(2*this.hx,1.2,2*this.hy);object.rotation.y=-this.angle;}
    this.paint(0xffcd77,'pending','正在核验位置…');
    this.host.dataset.wallAngle=String(this.angle);this.host.dataset.wallX=x.toFixed(3);this.host.dataset.wallY=y.toFixed(3);
    this.schedule();
  }

  refresh():void {if(this.point&&performance.now()-this.checkedAt>600)this.schedule();}

  hide():void {
    this.point=null;this.revision++;clearTimeout(this.timer);this.timer=0;
    this.mesh.visible=this.edges.visible=false;delete this.host.dataset.wallPreview;
    delete this.host.dataset.wallX;delete this.host.dataset.wallY;this.onStatus('');
  }

  private paint(color:number,state:string,message:string):void {
    this.material.color.setHex(color);this.edgeMaterial.color.setHex(color);
    this.host.dataset.wallPreview=state;this.onStatus(message);
  }

  private schedule():void {
    if(this.timer)return;
    this.timer=window.setTimeout(()=>{this.timer=0;void this.check();},100);
  }

  private async check():Promise<void> {
    const point=this.point;if(!point)return;
    const revision=this.revision;this.checkedAt=performance.now();
    const query=new URLSearchParams({x:String(point.x),y:String(point.y),hx:String(this.hx),hy:String(this.hy),angle:String(this.angle)});
    try {
      const response=await fetch(`/api/wall-preview?${query}`,{signal:AbortSignal.timeout(2000)});
      if(revision!==this.revision)return;
      if(!response.ok){this.paint(0xff7676,'invalid','超出场地或尺寸范围');return;}
      const placement=await response.json() as Placement;
      if(revision!==this.revision)return;
      this.paint(placement.valid?(placement.displaced?0xffcd77:0x66d9ef):0xff7676,placement.valid?'valid':'invalid',placement.message);
    } catch {
      if(revision===this.revision)this.paint(0xffcd77,'pending','等待本地校验；点击时再次检查');
    }
  }
}
