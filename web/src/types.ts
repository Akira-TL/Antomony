export interface AntState {
  id:number;x:number;y:number;heading:number;carrying:boolean;delivered:number;contacts:number;
  updates:number;frozen:boolean;error:number;delta:number;drift:number;fingerprint:string;
  birth_loss:number;warm_loss:number;hidden:number[];inputs:number[];prediction:number[];rays:number[];action:number;
  sense_x:number;sense_y:number;sense_heading:number;following_trail:boolean;
}
export interface WallState {id:number;x:number;y:number;hx:number;hy:number;angle:number}
export interface FoodState {id:number;x:number;y:number;amount:number}
export interface EventState {tick:number;kind:string;message:string;ant:number;value:number}
export interface Frame {
  mode:'neural'|'rules';distance:number;stalled:number;
  tick:number;seconds:number;seed:number;paused:boolean;rate:number;delivered:number;contacts:number;samples:number;
  mean_error:number;tick_ms:number;wind:number;field_enabled:boolean;field_width:number;field_height:number;pheromones:string;
  ants:AntState[];walls:WallState[];foods:FoodState[];events:EventState[];
}
export interface ComparisonFrame extends Frame {reference:Frame|null}
export type SceneFrame = Pick<Frame,'mode'|'tick'|'seconds'|'seed'|'delivered'|'field_width'|'field_height'|'pheromones'|'walls'|'foods'> & {
  ants:Pick<AntState,'id'|'x'|'y'|'heading'|'carrying'|'frozen'|'rays'|'sense_x'|'sense_y'|'sense_heading'>[];
};
export type Tool='inspect'|'wall'|'erase'|'food'|'scent';
export interface Command {
  kind:'pause'|'step'|'reset'|'speed'|'wall'|'erase'|'food'|'scent'|'clear'|'learning'|'freeze-ant'|'wind'|'fields'|'compare';
  x?:number;y?:number;value?:number;ant?:number;seed?:number;count?:number;hx?:number;hy?:number;angle?:number;
}
