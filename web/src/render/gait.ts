export interface Foot {x:number;y:number;z:number;moving:boolean}
interface Leg extends Foot {fromX:number;fromZ:number;toX:number;toZ:number;progress:number}

/** 支撑足固定在世界坐标；仅在伸展超过阈值且邻足允许时迈步。 */
export class AntGait {
  private legs:Leg[]=[];
  private previousX=0;
  private previousZ=0;

  reset(x:number,z:number,heading:number):void {
    this.legs=Array.from({length:6},(_,i)=>{
      const p=this.rest(i,x,z,heading);
      const phase=[.12,.06,0,.15,.09,.03][i];
      const foot={x:p.x+Math.cos(heading)*phase,z:p.z+Math.sin(heading)*phase};
      return {...foot,y:.025,moving:false,fromX:foot.x,fromZ:foot.z,toX:foot.x,toZ:foot.z,progress:1};
    });
    this.previousX=x;this.previousZ=z;
  }

  private rest(i:number,x:number,z:number,heading:number):{x:number;z:number}{
    const along=(i%3-1)*.14,side=i<3?-1:1,c=Math.cos(heading),s=Math.sin(heading);
    return {x:x+c*along-s*.25*side,z:z+s*along+c*.25*side};
  }

  private swingTarget(i:number,x:number,z:number,heading:number):{x:number;z:number}{
    const c=Math.cos(heading),s=Math.sin(heading),rest=this.rest(i,x,z,heading);
    const target={x:rest.x+c*.21,z:rest.z+s*.21};
    let along=target.x*c+target.z*s;
    const onSide=i%3;
    if(onSide>0){const rear=this.legs[i-1];along=Math.max(along,rear.x*c+rear.z*s+.025);}
    if(onSide<2){const front=this.legs[i+1];along=Math.min(along,front.x*c+front.z*s-.025);}
    const correction=along-(target.x*c+target.z*s);
    target.x+=c*correction;target.z+=s*correction;
    return target;
  }

  update(x:number,z:number,heading:number,dt:number):readonly Foot[]{
    if(!this.legs.length||Math.hypot(x-this.previousX,z-this.previousZ)>1.2)this.reset(x,z,heading);
    this.previousX=x;this.previousZ=z;
    if(dt<=0)return this.legs;
    const c=Math.cos(heading),s=Math.sin(heading);
    for(const [i,leg] of this.legs.entries()){
      if(!leg.moving)continue;
      const target=this.swingTarget(i,x,z,heading);
      leg.toX=target.x;leg.toZ=target.z;
      leg.progress=Math.min(1,leg.progress+dt/.09);
      const t=leg.progress;
      leg.x=leg.fromX+(leg.toX-leg.fromX)*t;
      leg.z=leg.fromZ+(leg.toZ-leg.fromZ)*t;
      leg.y=.025+.09*Math.sin(Math.PI*t);
      if(t>=1){leg.moving=false;leg.y=.025;}
    }
    const candidates=this.legs.map((leg,i)=>{
      const rest=this.rest(i,x,z,heading),dx=leg.x-rest.x,dz=leg.z-rest.z;
      const rear=-(dx*c+dz*s),lateral=Math.abs(-dx*s+dz*c);
      return {i,rest,stretch:Math.max(rear/.065,lateral/.14)};
    });
    candidates.sort((a,b)=>b.stretch-a.stretch);
    for(const item of candidates){
      const i=item.i,leg=this.legs[i];
      if(leg.moving||item.stretch<1)continue;
      const locked=this.legs.some((other,j)=>other.moving&&(Math.floor(i/3)===Math.floor(j/3)||i%3===j%3));
      if(locked)continue;
      leg.fromX=leg.x;leg.fromZ=leg.z;
      const target=this.swingTarget(i,x,z,heading);
      leg.toX=target.x;leg.toZ=target.z;
      leg.progress=0;leg.moving=true;
    }
    return this.legs;
  }
}
