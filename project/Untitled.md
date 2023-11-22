사진을 넣어요
그럼 1.png가 보통 담기죠?
이걸 자바 컨트롤러에서 담죠?
mvo.username이 있겟죠?
a@gmail.com이라고 할때
"a@gmail.com-KakaoTalk_20230829_160433709.png"
"a@gmail.com-KakaoTalk_20230829_160433709-400.png"
"a@gmail.com-KakaoTalk_20230829_160433709-200.png"


{1:"a@gmail.com-KakaoTalk_20230829_160433709.png"}


나중에 이걸 불러와서 쓸 때는
```
String photo1 = "a@gmail.com-KakaoTalk_20230829_160433709.png"
<img src="${photo1}">
```



```
@GetMapping("/profile")
   public String showProfilePage(Model model, HttpSession session) {
      System.out.println("마이페이지로 들어왔음.");
      // 사진 출력되는 곳
      Info userInfo = (Info) session.getAttribute("mvo");
      Map<Integer, String> photoMap = userInfo.getPhoto();
      List<String> imageDatas = new ArrayList<>();
      String imageData = null;
      String bucketName = "simkoong-s3";
      String base64Encoded = null;      
      if (photoMap != null) {
         for (int i = 1; i <= 4; i++) {
            String imagePath = photoMap.get(i);
            if (imagePath != null) {
               File file = new File(imagePath);
               String fileName = file.getName();
               try {
                   S3Object s3object = s3client.getObject(bucket![[Pasted image 20231122155943.jpg]]Name, fileName);
                   S3ObjectInputStream inputStream = s3object.getObjectContent();
                   byte[] bytes = IOUtils.toByteArray(inputStream);
                   base64Encoded = Base64.encodeBase64String(bytes);
                   imageDatas.add(base64Encoded);
               } catch (Exception e) {
                   // 파일이 존재하지 않을 때 빈 이미지 추가
                   base64Encoded = ""; // 빈 문자열 또는 기본 이미지 URL 설정
                   imageDatas.add(base64Encoded);
               }            
            }
         }
      }
      model.addAttribute("imageDatas", imageDatas);
      model.addAttribute("imageData", base64Encoded);

      return "profile";
   }
```