export type Arm='learned'|'skip'|'always'|'mlp'|'rules';
export interface Ant {
  observation:number[];active:boolean;move:boolean;turn:number;position:number[];heading:number;
  carrying:boolean;exploration_left:number;reserve_left:number;picked_up:boolean;delivered:boolean;
  budget_return:boolean;exhausted:boolean;killed:boolean;injury:number;reward:number;writes:number;
}
export interface Frame {tick:number;source_position:number[];source_active:boolean;ants:Ant[]}
export interface Result {seed:number;condition:string;arm:Arm;steps:number;deliveries:number;deaths:number;writes:number[];snapshots:number[]}
export interface Header {result:Result;food:number[];initial_positions:number[][];initial_headings:number[];nest_radius:number;signal_radius:number;contact_radius:number;source_strength:number}
export interface Update {tick:number;individual:number;prediction:number|null;eligible:boolean;accepted:boolean;changed:boolean;before:number[];after:number[]}
export interface WeightPoint {tick:number;values:number[]}
export interface WeightGroup {name:string;shape:number[];frozen:boolean;points:WeightPoint[]}
export interface Count {deliveries:number;pickups:number;deaths:number;exhausted:number;writes:number;reward:number}
export interface Tape {header:Header;frames:Frame[];updates:Update[];counts:Count[]}
export interface Catalog {execution:{plan:{seeds:number[];conditions:{name:string}[];environment:{horizon:number;ants:number;stock:number}}};summary:{development_continue:boolean;passing_seeds:number;passing_conditions:number}}

export function counts(frames:Frame[]):Count[] {
  const result:Count[]=[{deliveries:0,pickups:0,deaths:0,exhausted:0,writes:0,reward:0}];
  for(const frame of frames){const prior=result.at(-1)!;result.push({
    deliveries:prior.deliveries+frame.ants.filter(a=>a.delivered).length,
    pickups:prior.pickups+frame.ants.filter(a=>a.picked_up).length,
    deaths:frame.ants.filter(a=>a.killed).length,exhausted:frame.ants.filter(a=>a.exhausted).length,
    writes:frame.ants.reduce((n,a)=>n+a.writes,0),reward:prior.reward+frame.ants.reduce((n,a)=>n+a.reward,0),
  });}return result;
}
export function pointAt(points:WeightPoint[],tick:number):WeightPoint {
  let selected=points[0];for(const point of points){if(point.tick>tick)break;selected=point;}return selected;
}
export function jsonLines<T>(text:string):T[]{return text.trim()?text.trim().split('\n').map(line=>JSON.parse(line) as T):[];}
