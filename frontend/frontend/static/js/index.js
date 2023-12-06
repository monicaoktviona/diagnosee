function search() {
    $.ajax({
      method:'POST',
      url:"/search",
      data:{
        query:$('#query').val(),
      },
    });
  }