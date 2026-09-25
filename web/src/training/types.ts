export type Phase='motor'|'meta'|'autonomous';
export type GroupId='action'|'gate'|'query'|'key'|'rate';
export type Lesson='straight'|'turn'|'random';
export interface GroupState {id:GroupId;label:string;start:number;end:number;frozen:boolean;manual:boolean;reason:string}
export interface EpisodeRecord {episode:number;phase:Phase;lesson:Lesson;steps:number;reward:number;reached:boolean;reason:string;self_updates:number;outer_updates:number;checkpoint:string}
export interface TrainingState {
  session:string;paused:boolean;error:string;phase:Phase;lesson:Lesson;speed:number;tick:number;episode:number;steps:number;horizon:number;
  x:number;y:number;heading:number;target_x:number;target_y:number;distance:number;max_turn:number;
  move:boolean;turn:number;move_probability:number;write_probability:number;write_status:string;
  reward:number;total_reward:number;self_updates:number;outer_updates:number;
  inputs:string[];observation:number[];groups:GroupState[];weights:number[][];initial:number[][];self_delta:number[][];outer_delta:number[][];history:EpisodeRecord[];
}
export interface TrainingCommand {
  action:'play'|'pause'|'step'|'phase'|'lesson'|'freeze'|'freeze_all'|'speed'|'turn'|'reset';
  phase?:Phase;lesson?:Lesson;group?:GroupId;frozen?:boolean;speed?:1|4|16;max_turn?:number;
}
