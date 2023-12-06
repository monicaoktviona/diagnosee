function search() {
    $.ajax({
      method:'POST',
      url:"/frontend/search",
      data:{
        query:$('#input-group').val(),
      },
    });
  }