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
      return {...p,y:.025,moving:false,fromX:p.x,fromZ:p.z,toX:p.x,toZ:p.z,progress:1};
    });
    this.previousX=x;this.previousZ=z;
  }

  private rest(i:number,x:number,z:number,heading:number):{x:number;z:number}{
    const along=(i%3-1)*.14,side=i<3?-1:1,c=Math.cos(heading),s=Math.sin(heading);
    return {x:x+c*along-s*.25*side,z:z+s*along+c*.25*side};
  }

  update(x:number,z:number,heading:number,dt:number):readonly Foot[]{
    if(!this.legs.length||Math.hypot(x-this.previousX,z-this.previousZ)>1.2)this.reset(x,z,heading);
    this.previousX=x;this.previousZ=z;
    if(dt<=0)return this.legs;
    for(const leg of this.legs){
      if(!leg.moving)continue;
      leg.progress=Math.min(1,leg.progress+dt/.085);
      const t=leg.progress;
      leg.x=leg.fromX+(leg.toX-leg.fromX)*t;
      leg.z=leg.fromZ+(leg.toZ-leg.fromZ)*t;
      leg.y=.025+.065*Math.sin(Math.PI*t);
      if(t>=1){leg.moving=false;leg.y=.025;}
    }
    const candidates=this.legs.map((leg,i)=>({i,rest:this.rest(i,x,z,heading),distance:0}));
    for(const item of candidates){const leg=this.legs[item.i];item.distance=Math.hypot(leg.x-item.rest.x,leg.z-item.rest.z);}
    candidates.sort((a,b)=>b.distance-a.distance);
    for(const item of candidates){
      const i=item.i,leg=this.legs[i];
      if(leg.moving||item.distance<.14)continue;
      const locked=this.legs.some((other,j)=>other.moving&&(Math.floor(i/3)===Math.floor(j/3)||i%3===j%3));
      if(locked)continue;
      leg.fromX=leg.x;leg.fromZ=leg.z;
      leg.toX=item.rest.x+Math.cos(heading)*.08;
      leg.toZ=item.rest.z+Math.sin(heading)*.08;
      leg.progress=0;leg.moving=true;
    }
    return this.legs;
  }
}
