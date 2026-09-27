export type GroupKey='adaptive'|'mlp'|'rules';
export interface Wall {id:number;x:number;y:number;hx:number;hy:number;angle:number}
export interface Food {id:number;x:number;y:number;stock:number}
export interface Trap {id:number;x:number;y:number;born_at:number;radius:number;injury:number;speed_multiplier:number;period:number;motion_amplitude:number;motion_period:number;response:number[]}
export interface Ant {id:number;x:number;y:number;heading:number;carrying:boolean;pending:boolean;injury:number;exploration:number;reserve:number;delivered:number;deaths:number;exhaustions:number;revivals:number;writes:number;receptors:number[]}
export interface Counts {delivered:number;pickups:number;deaths:number;exhaustions:number;revivals:number;active:number;injury:number;decisions:number;accepted:number;writes:number;stock:number}
export interface World {key:GroupKey;counts:Counts;ants:Ant[];walls:Wall[];foods:Food[];traps:{spec:Trap;position:number[];active:boolean}[];field:string}
export interface Frame {run_id:string;tick:number;horizon:number;seed:number;paused:boolean;done:boolean;learning:boolean;rate:number;step_ms:number;checkpoint_tick:number;groups:World[];notices:{tick:number;message:string}[];error:string|null}
export interface Edit {kind:'wall'|'food'|'trap'|'erase-wall'|'erase-trap'|'clear-trails';x?:number;y?:number;hx?:number;hy?:number;angle?:number;stock?:number;identifier?:number;radius?:number;injury?:number;speed_multiplier?:number;period?:number;motion_amplitude?:number}
export interface Preview {valid:boolean;message:string;tick:number}
export interface Control {kind:'pause'|'step'|'speed'|'learning'|'checkpoint';enabled?:boolean;rate?:number}
export interface ParameterPoint {tick:number;values:number[]}
export interface ParameterModule {key:string;label:string;frozen:boolean;points:ParameterPoint[]}
export interface Parameters {run_id:string;tick:number;group:GroupKey;individual:number;modules:ParameterModule[]}
export type Tool='inspect'|'wall'|'food'|'trap'|'erase';
export const labels:Record<GroupKey,string>={adaptive:'自训练模型',mlp:'普通MLP',rules:'代码规则'};

export function pageUrl(path:string,pageHref=window.location.href):string{
  const base=new URL('.',pageHref);
  if(!path.startsWith('/api/')&&!path.startsWith('api/'))throw new Error('仅允许当前页面目录的接口资源');
  const target=new URL(path.startsWith('/')?path.slice(1):path,base);
  if(target.origin!==base.origin||!target.pathname.startsWith(`${base.pathname}api/`))throw new Error('仅允许当前页面目录的接口资源');
  return target.href;
}

export async function api<T>(path:string,body?:object):Promise<T>{
  const response=await fetch(pageUrl(`/api/${path}`),{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:undefined,body:body?JSON.stringify(body):undefined,signal:AbortSignal.timeout(20000)});
  if(!response.ok){
    const error:unknown=await response.json().catch(()=>null);
    const detail=error&&typeof error==='object'&&'detail'in error?error.detail:null;
    throw new Error(typeof detail==='string'?detail:`请求未完成（${response.status}）`);
  }
  return response.json() as Promise<T>;
}
