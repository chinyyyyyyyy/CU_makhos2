<template>
  <div id="app">
    <Board />  
  </div>
</template>

<script>
import Board from './components/Board.vue'

export default {
  name: 'App',
  components: {
    Board,
  }
  
}
</script>

<style>
#app{
  width: 800px;
  margin: 0 auto;
  border: 1px solid #000;
}
</style>
